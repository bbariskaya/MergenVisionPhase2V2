/*
 * M5.2 real GPU batching smoke test.
 *
 * Builds the same decode pipeline as decode_smoke, then feeds each mux batch
 * through TemporalFrameBatchAssembler and VideoFacePipeline.  Verifies that:
 *   - decoded/processed counts match requested frames
 *   - PTS is monotonic and not derived
 *   - detector batch inference produces faces on Friends.mp4
 *   - frame identity is preserved through batching
 *   - tracker identity survives detector batch boundaries
 *
 * This is the first concrete gate for M5.2; it intentionally does not yet write
 * the observation protobuf artifact (that comes in mv_video_worker).
 */

#include <gst/gst.h>
#include <gst/video/video.h>

#include "gstnvdsmeta.h"
#include "nvbufsurface.h"

#include "mv/video/batch_assembler.hpp"
#include "mv/video/detection_mapper.hpp"
#include "mv/video/retained_buffer_handle.hpp"
#include "mv/video/tracker_adapter.hpp"
#include "mv/video/video_face_pipeline.hpp"

#include <chrono>
#include <condition_variable>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

struct SmokeOptions {
    std::string video_path;
    int gpu_id = 0;
    int detector_batch_size = 8;
    int recognizer_batch_size = 32;
    int max_frames = 300;
    bool rgba_oracle = false;
};

struct SmokeStats {
    std::mutex mtx;
    int mux_buffers = 0;
    int decoded_frames = 0;
    int accepted_frames = 0;
    int limit_discarded = 0;
    int processed_frames = 0;
    int total_detections = 0;
    int tracked_observations = 0;
    int raw_tracks = 0;
    int inference_batches = 0;
    int partial_batches = 0;
    int batch_size_min = std::numeric_limits<int>::max();
    int batch_size_max = 0;
    int64_t first_pts_ns = -1;
    int64_t last_pts_ns = -1;
    bool got_nvm = false;
    bool got_meta = false;
    bool error = false;
    std::string error_message;

    uint64_t tracker_us = 0;
    uint64_t pipeline_preprocess_us = 0;
    uint64_t pipeline_engine_us = 0;
    uint64_t pipeline_postproc_us = 0;
    uint64_t pipeline_mapping_us = 0;
    uint64_t pipeline_calls = 0;
};

struct PipelineContext {
    GstElement* pipeline = nullptr;
    GstElement* demux = nullptr;
    GstElement* parser = nullptr;
    GstElement* decoder = nullptr;
    GstElement* mux = nullptr;
    GstPad* mux_sink_pad = nullptr;

    std::mutex mtx;
    std::condition_variable cv;
    bool demux_linked = false;
    bool error = false;
    std::string error_message;
};

struct ProbeState {
    SmokeOptions* opts = nullptr;
    SmokeStats* stats = nullptr;
    mergenvision::video::TemporalFrameBatchAssembler* assembler = nullptr;
    mergenvision::video::VideoFacePipeline* pipeline = nullptr;
    mergenvision::video::NaiveTracker* tracker = nullptr;
    int64_t presentation_counter = 0;
    int64_t mux_counter = 0;
    bool stop_requested = false;
};

static void set_stat_error(SmokeStats* stats, const std::string& msg) {
    std::lock_guard<std::mutex> lock(stats->mtx);
    if (!stats->error) {
        stats->error = true;
        stats->error_message = msg;
    }
}

static const char* color_format_name(int fmt) {
    switch (fmt) {
        case NVBUF_COLOR_FORMAT_GRAY8: return "GRAY8";
        case NVBUF_COLOR_FORMAT_YUV420: return "YUV420";
        case NVBUF_COLOR_FORMAT_YVU420: return "YVU420";
        case NVBUF_COLOR_FORMAT_NV12: return "NV12";
        case NVBUF_COLOR_FORMAT_NV12_ER: return "NV12_ER";
        case NVBUF_COLOR_FORMAT_NV21: return "NV21";
        case NVBUF_COLOR_FORMAT_NV21_ER: return "NV21_ER";
        case NVBUF_COLOR_FORMAT_YUV444: return "YUV444";
        case NVBUF_COLOR_FORMAT_RGBA: return "RGBA";
        case NVBUF_COLOR_FORMAT_BGRA: return "BGRA";
        case NVBUF_COLOR_FORMAT_ARGB: return "ARGB";
        case NVBUF_COLOR_FORMAT_ABGR: return "ABGR";
        case NVBUF_COLOR_FORMAT_RGB: return "RGB";
        case NVBUF_COLOR_FORMAT_BGR: return "BGR";
        case NVBUF_COLOR_FORMAT_NV12_10LE: return "NV12_10LE";
        case NVBUF_COLOR_FORMAT_NV12_12LE: return "NV12_12LE";
        case NVBUF_COLOR_FORMAT_YUV420_709: return "YUV420_709";
        case NVBUF_COLOR_FORMAT_NV12_709: return "NV12_709";
        case NVBUF_COLOR_FORMAT_NV12_709_ER: return "NV12_709_ER";
        case NVBUF_COLOR_FORMAT_YUV420_2020: return "YUV420_2020";
        case NVBUF_COLOR_FORMAT_NV12_2020: return "NV12_2020";
        case NVBUF_COLOR_FORMAT_RGBA_10_10_10_2_709: return "RGBA_10_10_10_2_709";
        case NVBUF_COLOR_FORMAT_BGRA_10_10_10_2_709: return "BGRA_10_10_10_2_709";
        default: return "UNKNOWN";
    }
}

static const char* mem_type_name(int mem) {
    switch (mem) {
        case NVBUF_MEM_DEFAULT: return "DEFAULT";
        case NVBUF_MEM_CUDA_PINNED: return "CUDA_PINNED";
        case NVBUF_MEM_CUDA_DEVICE: return "CUDA_DEVICE";
        case NVBUF_MEM_CUDA_UNIFIED: return "CUDA_UNIFIED";
        case NVBUF_MEM_SURFACE_ARRAY: return "SURFACE_ARRAY";
        case NVBUF_MEM_HANDLE: return "HANDLE";
        case NVBUF_MEM_SYSTEM: return "SYSTEM";
        case NVBUF_MEM_CUDA_ARRAY: return "CUDA_ARRAY";
        default: return "UNKNOWN";
    }
}

static void log_surface_contract(const NvBufSurface* surf) {
    if (!surf) {
        g_print("[surface contract] NULL surface\n");
        return;
    }
    g_print("[surface contract] batchSize=%u numFilled=%u memType=%s gpuId=%u\n",
            surf->batchSize, surf->numFilled,
            mem_type_name(static_cast<int>(surf->memType)),
            surf->gpuId);
    for (uint32_t i = 0; i < surf->batchSize; ++i) {
        const auto& sp = surf->surfaceList[i];
        g_print("[surface contract]  surf[%u] %dx%d colorFormat=%s(%d) pitch=%u layout=%d dataPtr=%p\n",
                i, sp.width, sp.height,
                color_format_name(static_cast<int>(sp.colorFormat)),
                static_cast<int>(sp.colorFormat),
                sp.pitch, static_cast<int>(sp.layout), sp.dataPtr);
        g_print("[surface contract]   planeParams num_planes=%u\n", sp.planeParams.num_planes);
        for (uint32_t p = 0; p < sp.planeParams.num_planes; ++p) {
            g_print("[surface contract]     plane[%u] w=%u h=%u pitch=%u offset=%u psize=%u bytesPerPix=%u\n",
                    p,
                    sp.planeParams.width[p], sp.planeParams.height[p],
                    sp.planeParams.pitch[p], sp.planeParams.offset[p],
                    sp.planeParams.psize[p], sp.planeParams.bytesPerPix[p]);
        }
    }
}

static void post_error(PipelineContext* ctx, const char* msg) {
    std::lock_guard<std::mutex> lock(ctx->mtx);
    ctx->error = true;
    ctx->error_message = msg;
    ctx->cv.notify_all();
}

static GstElement* make_element_checked(const gchar* factory, const gchar* name, PipelineContext* ctx) {
    GstElement* element = gst_element_factory_make(factory, name);
    if (!element) {
        char buf[256];
        std::snprintf(buf, sizeof(buf), "failed to create element %s", factory);
        post_error(ctx, buf);
    }
    return element;
}

static bool set_state_blocking(GstElement* element, GstState state, PipelineContext* ctx) {
    GstStateChangeReturn ret = gst_element_set_state(element, state);
    if (ret == GST_STATE_CHANGE_FAILURE) {
        post_error(ctx, "state change failed");
        return false;
    }
    if (ret == GST_STATE_CHANGE_ASYNC) {
        GstState current = GST_STATE_VOID_PENDING;
        GstState pending = GST_STATE_VOID_PENDING;
        ret = gst_element_get_state(element, &current, &pending, GST_CLOCK_TIME_NONE);
        if (ret == GST_STATE_CHANGE_FAILURE || current != state) {
            post_error(ctx, "async state change did not reach target state");
            return false;
        }
    }
    return true;
}

static bool link_static(GstElement* src, GstElement* sink, PipelineContext* ctx) {
    if (!gst_element_link(src, sink)) {
        post_error(ctx, "static link failed");
        return false;
    }
    return true;
}

static void on_demux_pad_added(GstElement* /*element*/, GstPad* pad, gpointer user_data) {
    auto* ctx = static_cast<PipelineContext*>(user_data);
    GstCaps* caps = gst_pad_get_current_caps(pad);
    if (!caps) caps = gst_pad_query_caps(pad, nullptr);
    GstStructure* structure = gst_caps_get_structure(caps, 0);
    const gchar* name = gst_structure_get_name(structure);
    const bool is_video = g_str_has_prefix(name, "video/");
    gst_caps_unref(caps);
    if (!is_video) {
        g_print("[real_batching_smoke] ignoring non-video pad: %s\n", name);
        return;
    }
    GstPad* parser_sink = gst_element_get_static_pad(ctx->parser, "sink");
    if (!parser_sink) {
        post_error(ctx, "parser has no sink pad");
        return;
    }
    GstPadLinkReturn ret = gst_pad_link(pad, parser_sink);
    gst_object_unref(parser_sink);
    if (GST_PAD_LINK_FAILED(ret)) {
        post_error(ctx, "demux->parser link failed");
        return;
    }
    {
        std::lock_guard<std::mutex> lock(ctx->mtx);
        ctx->demux_linked = true;
    }
    ctx->cv.notify_all();
}

static void process_complete_inference_batch(
    const mergenvision::video::InferenceFrameBatch& batch,
    ProbeState* ps,
    SmokeStats* stats) {
    auto frame_detections = ps->pipeline->infer_detector_batch(batch);
    auto m = ps->pipeline->metrics();

    auto t_tracker_start = std::chrono::steady_clock::now();
    // Tracker update must be strictly chronological and batch-boundary aware.
    for (auto& fd : frame_detections) {
        auto tracked = ps->tracker->update(
            fd.frame.presentation_index,
            fd.frame.pts_ns,
            fd.detections);

        {
            std::lock_guard<std::mutex> lock(stats->mtx);
            ++stats->processed_frames;
            stats->total_detections += static_cast<int>(fd.detections.size());
            stats->tracked_observations += static_cast<int>(tracked.size());
            if (stats->first_pts_ns < 0) stats->first_pts_ns = fd.frame.pts_ns;
            stats->last_pts_ns = fd.frame.pts_ns;
        }
    }
    auto t_tracker_end = std::chrono::steady_clock::now();
    uint64_t tracker_us = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(t_tracker_end - t_tracker_start).count());

    {
        std::lock_guard<std::mutex> lock(stats->mtx);
        stats->tracker_us += tracker_us;
        stats->pipeline_preprocess_us += m.preprocess_us;
        stats->pipeline_engine_us += m.engine_enqueue_us;
        stats->pipeline_postproc_us += m.postproc_us;
        stats->pipeline_mapping_us += m.mapping_us;
        stats->pipeline_calls += m.total_calls;
    }
}

static GstPadProbeReturn on_mux_src_buffer(GstPad* pad, GstPadProbeInfo* info, gpointer user_data) {
    auto* ps = static_cast<ProbeState*>(user_data);
    auto* stats = ps->stats;
    auto* assembler = ps->assembler;
    GstBuffer* buffer = GST_PAD_PROBE_INFO_BUFFER(info);
    if (!buffer || ps->stop_requested) {
        return GST_PAD_PROBE_OK;
    }

    static bool surface_contract_logged = false;

    NvDsBatchMeta* batch_meta = gst_buffer_get_nvds_batch_meta(buffer);
    if (!batch_meta) {
        return GST_PAD_PROBE_OK;
    }

    GstMapInfo map_info;
    if (!gst_buffer_map(buffer, &map_info, GST_MAP_READ)) {
        set_stat_error(stats, "failed to map mux buffer");
        return GST_PAD_PROBE_OK;
    }

    NvBufSurface* surface = reinterpret_cast<NvBufSurface*>(map_info.data);
    if (!surface) {
        gst_buffer_unmap(buffer, &map_info);
        set_stat_error(stats, "mux buffer has no NvBufSurface");
        return GST_PAD_PROBE_OK;
    }

    if (surface->memType == NVBUF_MEM_CUDA_DEVICE) {
        std::lock_guard<std::mutex> lock(stats->mtx);
        stats->got_nvm = true;
    }
    stats->got_meta = true;

    if (!surface_contract_logged) {
        surface_contract_logged = true;
        GstCaps* caps = gst_pad_get_current_caps(pad);
        if (caps) {
            gchar* s = gst_caps_to_string(caps);
            g_print("[surface contract] mux src caps: %s\n", s);
            g_free(s);
            gst_caps_unref(caps);
        }
        log_surface_contract(surface);
    }

    const uint64_t mux_seq = ps->mux_counter++;
    std::vector<mergenvision::video::FrameEnvelope> mux_frames;

    for (NvDsFrameMetaList* l = batch_meta->frame_meta_list; l != nullptr; l = l->next) {
        NvDsFrameMeta* fm = reinterpret_cast<NvDsFrameMeta*>(l->data);
        if (!fm || fm->batch_id >= surface->batchSize) continue;

        const NvBufSurfaceParams& sp = surface->surfaceList[fm->batch_id];

        mergenvision::video::FrameEnvelope env;
        env.presentation_index = static_cast<uint64_t>(ps->presentation_counter++);
        env.decoded_sequence = env.presentation_index;
        env.sampled_sequence = env.presentation_index;
        env.mux_batch_sequence = mux_seq;
        env.position_in_mux_batch = static_cast<uint32_t>(fm->batch_id);
        env.source_id = 0;
        env.pad_index = 0;
        env.nvds_frame_num = static_cast<uint64_t>(fm->frame_num);
        env.pts_ns = fm->buf_pts;
        env.duration_ns = -1;
        env.pts_derived = false;
        env.coded_width = sp.width;
        env.coded_height = sp.height;
        env.display_width = fm->source_frame_width ? fm->source_frame_width : sp.width;
        env.display_height = fm->source_frame_height ? fm->source_frame_height : sp.height;
        env.rotation_degrees = 0;

        env.device_view.width = sp.width;
        env.device_view.height = sp.height;
        env.device_view.display_width = env.display_width;
        env.device_view.display_height = env.display_height;
        env.device_view.pitch = sp.pitch;
        env.device_view.format = static_cast<int>(sp.colorFormat);
        env.device_view.data_ptr = sp.dataPtr;
        env.device_view.surface_index = static_cast<int>(fm->batch_id);

        env.device_view.num_planes = sp.planeParams.num_planes;
        if (sp.planeParams.num_planes >= 1) {
            env.device_view.plane_offset[0] = sp.planeParams.offset[0];
            env.device_view.plane_pitch[0] = sp.planeParams.pitch[0];
        }
        if (sp.planeParams.num_planes >= 2) {
            env.device_view.plane_offset[1] = sp.planeParams.offset[1];
            env.device_view.plane_pitch[1] = sp.planeParams.pitch[1];
        }

        env.owner = mergenvision::video::retain_buffer(buffer);

        mux_frames.push_back(std::move(env));
    }

    const int frames_in_buffer = static_cast<int>(mux_frames.size());

    // Clamp accepted frames to --max-frames exactly. Excess frames within the
    // last mux buffer are discarded and reported separately.
    int accepted_now = frames_in_buffer;
    {
        std::lock_guard<std::mutex> lock(stats->mtx);
        if (ps->opts->max_frames > 0) {
            int remaining = std::max(0, ps->opts->max_frames - stats->accepted_frames);
            if ((int)mux_frames.size() > remaining) {
                int drop = (int)mux_frames.size() - remaining;
                mux_frames.resize(static_cast<size_t>(remaining));
                stats->limit_discarded += drop;
                accepted_now = remaining;
            }
        }
    }

    // Feed the authoritative assembler; it may emit complete inference batches.
    try {
        auto complete_batches = assembler->push(std::move(mux_frames));
        for (auto& inf_batch : complete_batches) {
            process_complete_inference_batch(inf_batch, ps, stats);
        }

        {
            std::lock_guard<std::mutex> lock(stats->mtx);
            ++stats->mux_buffers;
            stats->decoded_frames += frames_in_buffer;
            stats->accepted_frames += accepted_now;
            stats->batch_size_min = std::min(stats->batch_size_min, frames_in_buffer);
            stats->batch_size_max = std::max(stats->batch_size_max, frames_in_buffer);
            if (ps->opts->max_frames > 0 && stats->accepted_frames >= ps->opts->max_frames) {
                ps->stop_requested = true;
            }
        }
    } catch (const std::exception& e) {
        gst_buffer_unmap(buffer, &map_info);
        set_stat_error(stats, std::string("assembler/pipeline error: ") + e.what());
        return GST_PAD_PROBE_OK;
    }

    gst_buffer_unmap(buffer, &map_info);

    return GST_PAD_PROBE_OK;
}

static bool parse_options(int argc, char** argv, SmokeOptions* opts) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--input" && i + 1 < argc) {
            opts->video_path = argv[++i];
        } else if (arg == "--gpu-id" && i + 1 < argc) {
            opts->gpu_id = std::atoi(argv[++i]);
        } else if (arg == "--detector-batch-size" && i + 1 < argc) {
            opts->detector_batch_size = std::atoi(argv[++i]);
        } else if (arg == "--recognizer-batch-size" && i + 1 < argc) {
            opts->recognizer_batch_size = std::atoi(argv[++i]);
        } else if (arg == "--max-frames" && i + 1 < argc) {
            opts->max_frames = std::atoi(argv[++i]);
        } else if (arg == "--all-frames") {
            opts->max_frames = 0;
        } else if (arg == "--rgba-oracle") {
            opts->rgba_oracle = true;
        } else if (arg == "--help" || arg == "-h") {
            std::fprintf(stdout,
                "usage: %s --input <video_path> [--max-frames N | --all-frames]\n"
                "       [--gpu-id ID] [--detector-batch-size N] [--recognizer-batch-size N]\n"
                "       [--rgba-oracle]\n",
                argv[0]);
            std::exit(0);
        } else {
            std::fprintf(stderr, "unknown option: %s\n", arg.c_str());
            return false;
        }
    }
    if (opts->video_path.empty()) {
        std::fprintf(stderr, "error: --input is required\n");
        return false;
    }
    if (opts->detector_batch_size <= 0 || opts->recognizer_batch_size <= 0) {
        std::fprintf(stderr, "error: batch sizes must be positive\n");
        return false;
    }
    return true;
}

} // namespace

int main(int argc, char** argv) {
    SmokeOptions opts;
    if (!parse_options(argc, argv, &opts)) {
        return 1;
    }

    setenv("USE_NEW_NVSTREAMMUX", "0", 1);
    gst_init(&argc, &argv);

    const std::string retina_engine_path =
        "backend/artifacts/engines/deepstream9/retinaface_r50_dynamic.bs1.opt8.max64.fp16.trt1014.engine";
    const std::string glint_engine_path =
        "backend/artifacts/engines/deepstream9/glintr100.bs1.opt8.max64.fp16.trt1014.engine";

    mergenvision::video::VideoFacePipeline face_pipeline;
    {
        std::string init_error;
        if (!face_pipeline.init(opts.gpu_id, retina_engine_path, glint_engine_path, &init_error)) {
            std::fprintf(stderr, "pipeline init failed: %s\n", init_error.c_str());
            return 1;
        }
    }

    mergenvision::video::TemporalFrameBatchAssembler assembler(
        static_cast<size_t>(opts.detector_batch_size), false);
    mergenvision::video::NaiveTracker tracker;
    SmokeStats stats;
    ProbeState ps;
    ps.opts = &opts;
    ps.stats = &stats;
    ps.assembler = &assembler;
    ps.pipeline = &face_pipeline;
    ps.tracker = &tracker;

    PipelineContext ctx;
    ctx.pipeline = gst_pipeline_new("real-batching-smoke");
    ctx.demux = make_element_checked("qtdemux", "demux", &ctx);
    ctx.parser = make_element_checked("h264parse", "parser", &ctx);
    ctx.decoder = make_element_checked("nvv4l2decoder", "decoder", &ctx);
    ctx.mux = make_element_checked("nvstreammux", "mux", &ctx);
    GstElement* streamdemux = make_element_checked("nvstreamdemux", "streamdemux", &ctx);
    GstElement* queue = make_element_checked("queue", "queue", &ctx);
    GstElement* sink = make_element_checked("fakesink", "sink", &ctx);

    if (!ctx.pipeline || !ctx.demux || !ctx.parser || !ctx.decoder || !ctx.mux ||
        !streamdemux || !queue || !sink) {
        return 1;
    }

    gst_bin_add_many(GST_BIN(ctx.pipeline),
        ctx.demux, ctx.parser, ctx.decoder, ctx.mux,
        streamdemux, queue, sink, nullptr);

    g_object_set(ctx.decoder,
        "gpu-id", opts.gpu_id,
        "cudadec-memtype", 0,
        "drop-frame-interval", 0,
        "skip-frames", 0,
        "num-extra-surfaces", 32,
        nullptr);

    g_object_set(ctx.mux,
        "batch-size", opts.detector_batch_size,
        "batched-push-timeout", 33333,
        "width", opts.rgba_oracle ? 640 : 1920,
        "height", opts.rgba_oracle ? 640 : 1080,
        "enable-padding", FALSE,
        "gpu-id", opts.gpu_id,
        "live-source", FALSE,
        "nvbuf-memory-type", 2,
        "num-surfaces-per-frame", 1,
        "buffer-pool-size", 128,
        "attach-sys-ts", FALSE,
        nullptr);

    g_object_set(sink, "sync", FALSE, "async", FALSE, "qos", FALSE, nullptr);

    if (!link_static(ctx.parser, ctx.decoder, &ctx)) return 1;
    if (!link_static(ctx.mux, streamdemux, &ctx)) return 1;
    if (!link_static(queue, sink, &ctx)) return 1;

    if (opts.rgba_oracle) {
        GstElement* nvvidconv = make_element_checked("nvvideoconvert", "nvvidconv", &ctx);
        GstElement* capsfilter = make_element_checked("capsfilter", "capsfilter", &ctx);
        if (!nvvidconv || !capsfilter) return 1;
        gst_bin_add_many(GST_BIN(ctx.pipeline), nvvidconv, capsfilter, nullptr);
        g_object_set(nvvidconv,
            "gpu-id", opts.gpu_id,
            "nvbuf-memory-type", 2,
            nullptr);
        GstCaps* rgba_caps = gst_caps_from_string("video/x-raw(memory:NVMM), format=RGBA");
        g_object_set(capsfilter, "caps", rgba_caps, nullptr);
        gst_caps_unref(rgba_caps);

        if (!gst_element_link(ctx.decoder, nvvidconv)) {
            std::fprintf(stderr, "decoder->nvvidconv link failed\n");
            return 1;
        }
        if (!gst_element_link(nvvidconv, capsfilter)) {
            std::fprintf(stderr, "nvvidconv->capsfilter link failed\n");
            return 1;
        }
        ctx.mux_sink_pad = gst_element_request_pad_simple(ctx.mux, "sink_0");
        if (!ctx.mux_sink_pad) {
            std::fprintf(stderr, "failed to request mux sink_0\n");
            return 1;
        }
        GstPad* capsfilter_src = gst_element_get_static_pad(capsfilter, "src");
        if (gst_pad_link(capsfilter_src, ctx.mux_sink_pad) != GST_PAD_LINK_OK) {
            std::fprintf(stderr, "capsfilter->mux link failed\n");
            return 1;
        }
        gst_object_unref(capsfilter_src);
    } else {
        ctx.mux_sink_pad = gst_element_request_pad_simple(ctx.mux, "sink_0");
        if (!ctx.mux_sink_pad) {
            std::fprintf(stderr, "failed to request mux sink_0\n");
            return 1;
        }
        GstPad* decoder_src = gst_element_get_static_pad(ctx.decoder, "src");
        if (gst_pad_link(decoder_src, ctx.mux_sink_pad) != GST_PAD_LINK_OK) {
            std::fprintf(stderr, "decoder->mux link failed\n");
            return 1;
        }
        gst_object_unref(decoder_src);
    }

    GstPad* streamdemux_src = gst_element_request_pad_simple(streamdemux, "src_0");
    GstPad* queue_sink = gst_element_get_static_pad(queue, "sink");
    if (streamdemux_src && queue_sink) {
        gst_pad_link(streamdemux_src, queue_sink);
        gst_object_unref(streamdemux_src);
        gst_object_unref(queue_sink);
    }

    GstPad* mux_src = gst_element_get_static_pad(ctx.mux, "src");
    if (!mux_src) {
        std::fprintf(stderr, "mux has no src pad\n");
        return 1;
    }
    gst_pad_add_probe(mux_src, GST_PAD_PROBE_TYPE_BUFFER, on_mux_src_buffer, &ps, nullptr);
    gst_object_unref(mux_src);

    GstElement* filesrc = gst_element_factory_make("filesrc", "source");
    if (!filesrc) {
        std::fprintf(stderr, "failed to create filesrc\n");
        return 1;
    }
    g_object_set(filesrc, "location", opts.video_path.c_str(), nullptr);
    gst_bin_add(GST_BIN(ctx.pipeline), filesrc);
    if (!link_static(filesrc, ctx.demux, &ctx)) return 1;

    g_signal_connect(ctx.demux, "pad-added", G_CALLBACK(on_demux_pad_added), &ctx);

    g_print("[real_batching_smoke] starting: detector_batch=%d max_frames=%d\n",
            opts.detector_batch_size, opts.max_frames);

    {
        std::unique_lock<std::mutex> lock(ctx.mtx);
        if (!set_state_blocking(ctx.pipeline, GST_STATE_PLAYING, &ctx)) {
            std::fprintf(stderr, "failed to start pipeline: %s\n", ctx.error_message.c_str());
            return 1;
        }
        if (!ctx.demux_linked && !ctx.error) {
            ctx.cv.wait_for(lock, std::chrono::seconds(5),
                [&] { return ctx.demux_linked || ctx.error; });
        }
        if (ctx.error) {
            std::fprintf(stderr, "pipeline error: %s\n", ctx.error_message.c_str());
            return 1;
        }
        if (!ctx.demux_linked) {
            std::fprintf(stderr, "timeout waiting for demux video pad\n");
            return 1;
        }
    }

    const auto t_start = std::chrono::steady_clock::now();
    GstBus* bus = gst_element_get_bus(ctx.pipeline);
    bool eos = false;
    bool bus_error = false;

    while (!eos && !bus_error && !ps.stop_requested) {
        GstMessage* msg = gst_bus_timed_pop_filtered(
            bus, 5 * GST_SECOND,
            static_cast<GstMessageType>(GST_MESSAGE_ERROR | GST_MESSAGE_EOS));
        if (!msg) {
            std::lock_guard<std::mutex> lock(stats.mtx);
            g_print("[real_batching_smoke] progress: decoded=%d processed=%d detections=%d\n",
                    stats.decoded_frames, stats.processed_frames, stats.total_detections);
            continue;
        }
        switch (GST_MESSAGE_TYPE(msg)) {
        case GST_MESSAGE_EOS:
            eos = true;
            break;
        case GST_MESSAGE_ERROR: {
            GError* err = nullptr;
            gchar* dbg = nullptr;
            gst_message_parse_error(msg, &err, &dbg);
            std::fprintf(stderr, "bus error: %s\n", err ? err->message : "unknown");
            if (dbg) g_free(dbg);
            if (err) g_error_free(err);
            bus_error = true;
            break;
        }
        default:
            break;
        }
        gst_message_unref(msg);
    }

    // Flush any partial batch at EOS.
    auto final_batch = assembler.flush_eos();
    if (final_batch) {
        process_complete_inference_batch(final_batch.value(), &ps, &stats);
        {
            std::lock_guard<std::mutex> lock(stats.mtx);
            ++stats.partial_batches;
        }
    }

    gst_element_set_state(ctx.pipeline, GST_STATE_NULL);
    gst_object_unref(bus);
    if (ctx.mux_sink_pad) {
        gst_element_release_request_pad(ctx.mux, ctx.mux_sink_pad);
        gst_object_unref(ctx.mux_sink_pad);
    }
    gst_object_unref(ctx.pipeline);

    const auto t_end = std::chrono::steady_clock::now();
    const double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    bool final_error = false;
    std::string final_error_message;
    int final_decoded = 0;
    int final_processed = 0;
    int final_mux_buffers = 0;
    int final_detections = 0;
    int final_tracked = 0;
    int final_raw_tracks = 0;
    int final_partial_batches = 0;
    bool final_got_nvm = false;
    bool final_got_meta = false;
    int64_t final_first_pts = -1;
    int64_t final_last_pts = -1;
    uint64_t final_tracker_us = 0;
    uint64_t final_pipeline_preprocess_us = 0;
    uint64_t final_pipeline_engine_us = 0;
    uint64_t final_pipeline_postproc_us = 0;
    uint64_t final_pipeline_mapping_us = 0;
    uint64_t final_pipeline_calls = 0;

    {
        std::lock_guard<std::mutex> lock(stats.mtx);
        final_error = stats.error;
        final_error_message = stats.error_message;
        final_decoded = stats.decoded_frames;
        final_processed = stats.processed_frames;
        final_mux_buffers = stats.mux_buffers;
        final_detections = stats.total_detections;
        final_tracked = stats.tracked_observations;
        final_partial_batches = stats.partial_batches;
        final_got_nvm = stats.got_nvm;
        final_got_meta = stats.got_meta;
        final_first_pts = stats.first_pts_ns;
        final_last_pts = stats.last_pts_ns;
        final_raw_tracks = static_cast<int>(tracker.tracks().size());
        final_tracker_us = stats.tracker_us;
        final_pipeline_preprocess_us = stats.pipeline_preprocess_us;
        final_pipeline_engine_us = stats.pipeline_engine_us;
        final_pipeline_postproc_us = stats.pipeline_postproc_us;
        final_pipeline_mapping_us = stats.pipeline_mapping_us;
        final_pipeline_calls = stats.pipeline_calls;
    }

    const double fps = (elapsed_ms > 0.0 && final_processed > 0)
        ? (static_cast<double>(final_processed) * 1000.0 / elapsed_ms)
        : 0.0;

    g_print("\n=== real batching smoke summary ===\n");
    g_print("decoded frames: %d\n", final_decoded);
    g_print("processed frames: %d\n", final_processed);
    g_print("mux buffers: %d\n", final_mux_buffers);
    g_print("detections: %d\n", final_detections);
    g_print("tracked observations: %d\n", final_tracked);
    g_print("raw tracks: %d\n", final_raw_tracks);
    g_print("partial batches: %d\n", final_partial_batches);
    g_print("NVMM device memory: %s\n", final_got_nvm ? "yes" : "no");
    g_print("PTS range: %" G_GINT64_FORMAT " .. %" G_GINT64_FORMAT " ns\n",
            final_first_pts, final_last_pts);
    g_print("wall time: %.2f ms\n", elapsed_ms);
    g_print("pipeline fps: %.2f\n", fps);
    if (final_pipeline_calls > 0) {
        auto avg = [&](uint64_t us) { return static_cast<double>(us) / static_cast<double>(final_pipeline_calls); };
        g_print("per-call avg (us): preprocess=%.0f enqueue=%.0f postproc=%.0f mapping=%.0f tracker=%.0f\n",
                avg(final_pipeline_preprocess_us),
                avg(final_pipeline_engine_us),
                avg(final_pipeline_postproc_us),
                avg(final_pipeline_mapping_us),
                avg(final_tracker_us));
    }

    bool pass = true;
    if (final_error) {
        std::fprintf(stderr, "runtime error: %s\n", final_error_message.c_str());
        pass = false;
    }
    if (!final_got_nvm || !final_got_meta) {
        std::fprintf(stderr, "missing NVMM stream metadata\n");
        pass = false;
    }
    if (opts.max_frames > 0 && final_processed != opts.max_frames) {
        std::fprintf(stderr, "frame count mismatch: expected %d, got %d\n",
                     opts.max_frames, final_processed);
        pass = false;
    }
    if (final_detections == 0) {
        std::fprintf(stderr, "no detections produced\n");
        pass = false;
    }
    if (final_tracked == 0) {
        std::fprintf(stderr, "no tracked observations produced\n");
        pass = false;
    }
    if (final_raw_tracks == 0) {
        std::fprintf(stderr, "no raw tracks produced\n");
        pass = false;
    }

    if (pass) {
        g_print("real batching smoke PASSED\n");
        return 0;
    }
    std::fprintf(stderr, "real batching smoke FAILED\n");
    return 1;
}

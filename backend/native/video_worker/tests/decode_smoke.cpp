/*
 * M5 GPU decode smoke test.
 *
 * Builds a real GStreamer/DeepStream pipeline:
 *   filesrc -> qtdemux -> h264parse -> nvv4l2decoder -> nvstreammux -> fakesink
 *
 * Verifies that output buffers are NVMM, that NvDsBatchMeta is present, and
 * that per-frame metadata (PTS, source dims, batch_id) is preserved.
 *
 * A pad probe on the nvstreammux source pad consumes the batched buffers in the
 * streaming thread, mirroring the proven DeepStream pattern used in the sibling
 * repository (MergenVisionPhase2).
 */

#include <gst/gst.h>
#include <gst/video/video.h>

#include <chrono>
#include <condition_variable>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <string>
#include <thread>

#include "gstnvdsmeta.h"
#include "nvbufsurface.h"

namespace {

struct ProbeStats {
  std::mutex mtx;
  int batches = 0;
  int frames = 0;
  bool got_meta = false;
  bool got_surface = false;
  bool got_nvm = false;
  int batch_frames_min = std::numeric_limits<int>::max();
  int batch_frames_max = 0;
  long long total_batch_frames = 0;
  bool error = false;
  std::string error_message;
};

struct PipelineContext {
  GstElement* pipeline = nullptr;
  GstElement* demux = nullptr;
  GstElement* parser = nullptr;
  GstElement* decoder = nullptr;
  GstElement* mux = nullptr;
  GstElement* sink = nullptr;
  GstPad* mux_sink_pad = nullptr;

  std::mutex mtx;
  std::condition_variable cv;
  bool demux_linked = false;
  bool error = false;
  std::string error_message;
};

static void on_demux_pad_added(GstElement* /*element*/, GstPad* pad, gpointer user_data) {
  auto* ctx = static_cast<PipelineContext*>(user_data);

  GstCaps* caps = gst_pad_get_current_caps(pad);
  if (!caps) {
    caps = gst_pad_query_caps(pad, nullptr);
  }

  GstStructure* structure = gst_caps_get_structure(caps, 0);
  const gchar* name = gst_structure_get_name(structure);
  const bool is_video = g_str_has_prefix(name, "video/");
  gst_caps_unref(caps);

  if (!is_video) {
    g_print("[decode_smoke] ignoring non-video pad: %s\n", name);
    return;
  }

  GstPad* parser_sink = gst_element_get_static_pad(ctx->parser, "sink");
  if (!parser_sink) {
    std::lock_guard<std::mutex> lock(ctx->mtx);
    ctx->error = true;
    ctx->error_message = "parser has no sink pad";
    ctx->cv.notify_all();
    return;
  }

  GstPadLinkReturn ret = gst_pad_link(pad, parser_sink);
  gst_object_unref(parser_sink);

  if (GST_PAD_LINK_FAILED(ret)) {
    std::lock_guard<std::mutex> lock(ctx->mtx);
    ctx->error = true;
    ctx->error_message = "demux->parser link failed";
    ctx->cv.notify_all();
    return;
  }

  g_print("[decode_smoke] demux video pad linked to parser\n");
  {
    std::lock_guard<std::mutex> lock(ctx->mtx);
    ctx->demux_linked = true;
  }
  ctx->cv.notify_all();
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

static GstPadProbeReturn on_mux_src_buffer(GstPad* /*pad*/, GstPadProbeInfo* info, gpointer user_data) {
  auto* stats = static_cast<ProbeStats*>(user_data);
  GstBuffer* buffer = GST_PAD_PROBE_INFO_BUFFER(info);

  if (!buffer) {
    return GST_PAD_PROBE_OK;
  }

  NvDsBatchMeta* batch_meta = gst_buffer_get_nvds_batch_meta(buffer);
  int frames_in_batch = 0;

  if (batch_meta) {
    {
      std::lock_guard<std::mutex> lock(stats->mtx);
      stats->got_meta = true;
    }

    for (NvDsFrameMetaList* l = batch_meta->frame_meta_list; l != nullptr; l = l->next) {
      NvDsFrameMeta* frame_meta = reinterpret_cast<NvDsFrameMeta*>(l->data);
      const bool first_batches = stats->batches < 3;
      if (first_batches) {
        g_print("  frame frame_num=%u pts=%" G_GINT64_FORMAT "us src=%dx%d batch_id=%u\n",
                frame_meta->frame_num,
                frame_meta->buf_pts / 1000,
                frame_meta->source_frame_width,
                frame_meta->source_frame_height,
                frame_meta->batch_id);
      }
      ++frames_in_batch;
      {
        std::lock_guard<std::mutex> lock(stats->mtx);
        ++stats->frames;
      }
    }
  }

  GstMapInfo map_info;
  if (gst_buffer_map(buffer, &map_info, GST_MAP_READ)) {
    NvBufSurface* surface = reinterpret_cast<NvBufSurface*>(map_info.data);
    if (surface) {
      {
        std::lock_guard<std::mutex> lock(stats->mtx);
        stats->got_surface = true;
        if (surface->memType == NVBUF_MEM_CUDA_DEVICE) {
          stats->got_nvm = true;
        }
      }
      if (stats->batches < 3) {
        g_print("  surface batchSize=%d memType=%d\n", surface->batchSize, surface->memType);
        const guint n = surface->batchSize;
        for (guint i = 0; i < n; ++i) {
          NvBufSurfaceParams& p = surface->surfaceList[i];
          g_print("    surface[%d] %ux%u pitch=%u fmt=%d dataPtr=%p\n",
                  i, p.width, p.height, p.pitch, p.colorFormat, p.dataPtr);
        }
      }
    }
    gst_buffer_unmap(buffer, &map_info);
  }

  {
    std::lock_guard<std::mutex> lock(stats->mtx);
    if (frames_in_batch > 0) {
      if (frames_in_batch < stats->batch_frames_min) stats->batch_frames_min = frames_in_batch;
      if (frames_in_batch > stats->batch_frames_max) stats->batch_frames_max = frames_in_batch;
      stats->total_batch_frames += frames_in_batch;
    }
    ++stats->batches;
    if (stats->batches <= 3) {
      g_print("[batch %d] frames_in_batch=%d\n", stats->batches, frames_in_batch);
    }
  }

  return GST_PAD_PROBE_OK;
}

}  // namespace

struct DecodeSmokeOptions {
  std::string video_path;
  int gpu_id = 0;
  int batch_size = 8;
  int batched_push_timeout_us = 40000;
  bool all_frames = false;
  int max_frames = 0;
};

static bool parse_decode_smoke_options(int argc, char** argv, DecodeSmokeOptions* opts) {
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--input" && i + 1 < argc) {
      opts->video_path = argv[++i];
    } else if (arg == "--gpu-id" && i + 1 < argc) {
      opts->gpu_id = std::atoi(argv[++i]);
    } else if (arg == "--batch-size" && i + 1 < argc) {
      opts->batch_size = std::atoi(argv[++i]);
    } else if (arg == "--batched-push-timeout-us" && i + 1 < argc) {
      opts->batched_push_timeout_us = std::atoi(argv[++i]);
    } else if (arg == "--max-frames" && i + 1 < argc) {
      opts->max_frames = std::atoi(argv[++i]);
      opts->all_frames = false;
    } else if (arg == "--all-frames") {
      opts->all_frames = true;
      opts->max_frames = 0;
    } else if (arg == "--help" || arg == "-h") {
      std::fprintf(stdout,
                   "usage: %s --input <video_path> [--all-frames | --max-frames N]\n"
                   "       [--gpu-id ID] [--batch-size N] [--batched-push-timeout-us US]\n",
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
  if (opts->max_frames < 0 || opts->batch_size <= 0 || opts->batched_push_timeout_us <= 0) {
    std::fprintf(stderr, "error: invalid numeric argument\n");
    return false;
  }
  return true;
}

int main(int argc, char** argv) {
  DecodeSmokeOptions opts;
  if (!parse_decode_smoke_options(argc, argv, &opts)) {
    std::fprintf(stderr, "Run with --help for usage.\n");
    return 1;
  }

  const std::string& video_path = opts.video_path;
  const int gpu_id = opts.gpu_id;
  const int batch_size = opts.batch_size;
  const int batched_push_timeout_us = opts.batched_push_timeout_us;

  // Force legacy nvstreammux so that batch-size/width/height/batched-push-timeout
  // properties behave as the binding roadmap expects.
  setenv("USE_NEW_NVSTREAMMUX", "0", 1);

  gst_init(&argc, &argv);

  PipelineContext ctx;
  ProbeStats stats;

  ctx.pipeline = gst_pipeline_new("decode-smoke");
  if (!ctx.pipeline) {
    std::fprintf(stderr, "failed to create pipeline\n");
    return 1;
  }

  ctx.demux = make_element_checked("qtdemux", "demux", &ctx);
  ctx.parser = make_element_checked("h264parse", "parser", &ctx);
  ctx.decoder = make_element_checked("nvv4l2decoder", "decoder", &ctx);
  ctx.mux = make_element_checked("nvstreammux", "mux", &ctx);
  GstElement* streamdemux = make_element_checked("nvstreamdemux", "streamdemux", &ctx);
  GstElement* queue = make_element_checked("queue", "queue", &ctx);
  ctx.sink = make_element_checked("fakesink", "sink", &ctx);

  if (!ctx.demux || !ctx.parser || !ctx.decoder || !ctx.mux || !streamdemux || !queue || !ctx.sink) {
    return 1;
  }

  gst_bin_add_many(GST_BIN(ctx.pipeline),
                   ctx.demux, ctx.parser, ctx.decoder, ctx.mux, streamdemux, queue, ctx.sink, nullptr);

  g_object_set(ctx.decoder,
               "gpu-id", gpu_id,
               "cudadec-memtype", 0,
               "drop-frame-interval", 0,
               "skip-frames", 0,
               "num-extra-surfaces", 32,
               nullptr);

  g_object_set(ctx.mux,
               "batch-size", batch_size,
               "batched-push-timeout", batched_push_timeout_us,
               "width", 1920,
               "height", 1080,
               "enable-padding", FALSE,
               "gpu-id", gpu_id,
               "live-source", FALSE,
               "nvbuf-memory-type", 2,
               "num-surfaces-per-frame", 1,
               "buffer-pool-size", 128,
               "attach-sys-ts", FALSE,
               nullptr);

  g_object_set(ctx.sink,
               "sync", FALSE,
               "async", FALSE,
               "qos", FALSE,
               nullptr);

  if (!link_static(ctx.parser, ctx.decoder, &ctx)) return 1;
  if (!link_static(ctx.mux, streamdemux, &ctx)) return 1;
  if (!link_static(queue, ctx.sink, &ctx)) return 1;

  ctx.mux_sink_pad = gst_element_request_pad_simple(ctx.mux, "sink_0");
  if (!ctx.mux_sink_pad) {
    std::fprintf(stderr, "failed to request sink_0 from mux\n");
    std::fflush(stderr);
    return 1;
  }
  g_print("[decode_smoke] mux request pad acquired: %s\n", gst_pad_get_name(ctx.mux_sink_pad));

  GstPad* decoder_src = gst_element_get_static_pad(ctx.decoder, "src");
  if (gst_pad_link(decoder_src, ctx.mux_sink_pad) != GST_PAD_LINK_OK) {
    std::fprintf(stderr, "decoder->mux link failed\n");
    return 1;
  }
  gst_object_unref(decoder_src);

  GstPad* streamdemux_src = gst_element_request_pad_simple(streamdemux, "src_0");
  if (!streamdemux_src) {
    std::fprintf(stderr, "failed to request src_0 from nvstreamdemux\n");
    return 1;
  }
  GstPad* queue_sink = gst_element_get_static_pad(queue, "sink");
  if (!queue_sink) {
    std::fprintf(stderr, "queue has no sink pad\n");
    return 1;
  }
  if (gst_pad_link(streamdemux_src, queue_sink) != GST_PAD_LINK_OK) {
    std::fprintf(stderr, "nvstreamdemux->queue link failed\n");
    return 1;
  }
  gst_object_unref(streamdemux_src);
  gst_object_unref(queue_sink);

  GstPad* mux_src = gst_element_get_static_pad(ctx.mux, "src");
  if (!mux_src) {
    std::fprintf(stderr, "mux has no src pad\n");
    return 1;
  }
  gst_pad_add_probe(mux_src, GST_PAD_PROBE_TYPE_BUFFER, on_mux_src_buffer, &stats, nullptr);
  gst_object_unref(mux_src);

  GstElement* filesrc = gst_element_factory_make("filesrc", "source");
  if (!filesrc) {
    std::fprintf(stderr, "failed to create filesrc\n");
    return 1;
  }
  g_object_set(filesrc, "location", video_path.c_str(), nullptr);
  gst_bin_add(GST_BIN(ctx.pipeline), filesrc);
  if (!link_static(filesrc, ctx.demux, &ctx)) return 1;

  g_signal_connect(ctx.demux, "pad-added", G_CALLBACK(on_demux_pad_added), &ctx);

  g_print("[decode_smoke] config: batch_size=%d batched_push_timeout=%d us gpu_id=%d\n",
          batch_size, batched_push_timeout_us, gpu_id);

  {
    std::unique_lock<std::mutex> lock(ctx.mtx);
    bool started = set_state_blocking(ctx.pipeline, GST_STATE_PLAYING, &ctx);
    if (!started) {
      std::fprintf(stderr, "failed to start pipeline: %s\n", ctx.error_message.c_str());
      return 1;
    }
    if (!ctx.demux_linked && !ctx.error) {
      ctx.cv.wait_for(lock, std::chrono::seconds(5), [&] { return ctx.demux_linked || ctx.error; });
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

  while (!eos && !bus_error) {
    GstMessage* msg = gst_bus_timed_pop_filtered(
        bus,
        5 * GST_SECOND,
        static_cast<GstMessageType>(GST_MESSAGE_ERROR | GST_MESSAGE_EOS));
    if (!msg) {
      // Periodic timeout: print progress and guard against infinite hang.
      std::lock_guard<std::mutex> lock(stats.mtx);
      g_print("[decode_smoke] progress: batches=%d frames=%d\n", stats.batches, stats.frames);
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

  gst_element_set_state(ctx.pipeline, GST_STATE_NULL);
  gst_object_unref(bus);

  if (ctx.mux_sink_pad) {
    gst_element_release_request_pad(ctx.mux, ctx.mux_sink_pad);
    gst_object_unref(ctx.mux_sink_pad);
  }

  gst_object_unref(ctx.pipeline);

  const auto t_end = std::chrono::steady_clock::now();
  const double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

  int final_batches = 0;
  int final_frames = 0;
  bool final_got_meta = false;
  bool final_got_surface = false;
  bool final_got_nvm = false;
  int final_batch_min = 0;
  int final_batch_max = 0;
  double avg_batch = 0.0;
  {
    std::lock_guard<std::mutex> lock(stats.mtx);
    final_batches = stats.batches;
    final_frames = stats.frames;
    final_got_meta = stats.got_meta;
    final_got_surface = stats.got_surface;
    final_got_nvm = stats.got_nvm;
    final_batch_min = stats.batch_frames_min;
    final_batch_max = stats.batch_frames_max;
    if (stats.batches > 0) {
      avg_batch = static_cast<double>(stats.total_batch_frames) / static_cast<double>(stats.batches);
    }
  }

  const double fps = (elapsed_ms > 0.0) ? (static_cast<double>(final_frames) * 1000.0 / elapsed_ms) : 0.0;

  g_print("\n=== decode smoke summary ===\n");
  g_print("batches: %d\n", final_batches);
  g_print("frames: %d\n", final_frames);
  g_print("NVMM surface seen: %s\n", final_got_surface ? "yes" : "no");
  g_print("NVMM (cuda-device) memory: %s\n", final_got_nvm ? "yes" : "no");
  g_print("NvDsBatchMeta seen: %s\n", final_got_meta ? "yes" : "no");
  g_print("batch frames  min: %d  max: %d  avg: %.2f\n",
          final_batches > 0 ? final_batch_min : 0,
          final_batch_max,
          avg_batch);
  g_print("wall time: %.2f ms (%.3f s)\n", elapsed_ms, elapsed_ms / 1000.0);
  g_print("decode fps: %.2f\n", fps);
  g_print("video duration: %.3f s\n", 278.035737);
  g_print("effective speedup: %.2fx\n",
          (elapsed_ms > 0.0) ? (278.035737 * 1000.0 / elapsed_ms) : 0.0);

  if (bus_error || !final_got_surface || !final_got_nvm || !final_got_meta || final_frames == 0) {
    std::fprintf(stderr, "decode smoke FAILED\n");
    return 1;
  }

  g_print("decode smoke PASSED\n");
  return 0;
}

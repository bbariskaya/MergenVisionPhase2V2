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

#include <algorithm>
#include <unordered_map>
#include <gst/gst.h>
#include <gst/video/video.h>

#include "gstnvdsmeta.h"
#include "nvbufsurface.h"

#include "mv/video/batch_assembler.hpp"
#include "mv/video/detection_mapper.hpp"
#include "mv/video/retained_buffer_handle.hpp"
#include "mv/video/tracker_adapter.hpp"
#include "mv/video/video_face_pipeline.hpp"

#ifdef MV_VIDEO_WORKER
#include "video_observation_v1.pb.h"
#include "video_track_template_v1.pb.h"
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/util/delimited_message_util.h>
#include <NvInfer.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <nvml.h>
#include <webp/encode.h>
#include <zstd.h>
#endif

#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {

struct SmokeOptions {
    std::string video_path;
    int gpu_id = 0;
    int detector_batch_size = 16;
    int recognizer_batch_size = 32;
    int max_frames = 300;
#ifdef MV_VIDEO_WORKER
    std::string output_dir;
    std::string job_id;
    std::string video_id;
    std::string model_profile;
    std::string detector_engine;
    std::string recognizer_engine;
#endif
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
    uint64_t pipeline_recognition_us = 0;
    uint64_t pipeline_mapping_us = 0;
    uint64_t pipeline_calls = 0;
    int total_embeddings = 0;
    int embedding_dim_errors = 0;
    int embedding_finite_errors = 0;
    double embedding_norm_min = 0.0;
    double embedding_norm_max = 0.0;
};

struct WorkQueue {
    std::mutex mtx;
    std::condition_variable cv;
    std::deque<std::vector<mergenvision::video::FrameEnvelope>> items;
    bool eos = false;
    bool error = false;
    std::string error_message;

    void push(std::vector<mergenvision::video::FrameEnvelope> frames) {
        {
            std::lock_guard<std::mutex> lock(mtx);
            if (eos) return;
            items.emplace_back(std::move(frames));
        }
        cv.notify_one();
    }

    void set_error(const std::string& msg) {
        {
            std::lock_guard<std::mutex> lock(mtx);
            if (!error) {
                error = true;
                error_message = msg;
            }
            eos = true;
        }
        cv.notify_all();
    }

    void set_eos() {
        {
            std::lock_guard<std::mutex> lock(mtx);
            eos = true;
        }
        cv.notify_all();
    }

    std::optional<std::vector<mergenvision::video::FrameEnvelope>> pop() {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [&] { return !items.empty() || eos; });
        if (items.empty()) return std::nullopt;
        auto front = std::move(items.front());
        items.pop_front();
        return front;
    }
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

#ifdef MV_VIDEO_WORKER
struct ArtifactState;
#endif

struct ProbeState {
    SmokeOptions* opts = nullptr;
    SmokeStats* stats = nullptr;
    WorkQueue* queue = nullptr;
    mergenvision::video::TemporalFrameBatchAssembler* assembler = nullptr;
    mergenvision::video::VideoFacePipeline* pipeline = nullptr;
    mergenvision::video::NaiveTracker* tracker = nullptr;
    int64_t presentation_counter = 0;
    int64_t mux_counter = 0;
    bool stop_requested = false;
#ifdef MV_VIDEO_WORKER
    ArtifactState* artifact = nullptr;
#endif
};

#ifdef MV_VIDEO_WORKER

using AlignedCropBuffer = mergenvision::video::AlignedCropBuffer;

static void delete_file_silent(const std::filesystem::path& p) {
    std::error_code ec;
    std::filesystem::remove(p, ec);
}

static void compress_file_zstd(const std::filesystem::path& src,
                               const std::filesystem::path& dst) {
    std::ifstream in(src, std::ios::binary);
    if (!in) {
        throw std::runtime_error("cannot open source file for zstd compression: " + src.string());
    }
    in.seekg(0, std::ios::end);
    const std::streamoff src_size_off = in.tellg();
    in.seekg(0, std::ios::beg);
    if (src_size_off < 0) {
        throw std::runtime_error("cannot determine source file size: " + src.string());
    }
    const size_t src_size = static_cast<size_t>(src_size_off);
    std::vector<char> input(src_size);
    in.read(input.data(), static_cast<std::streamsize>(src_size));
    if (!in) {
        throw std::runtime_error("failed to read source file for zstd compression: " + src.string());
    }

    const size_t bound = ZSTD_compressBound(src_size);
    std::vector<uint8_t> output(bound);
    const size_t compressed = ZSTD_compress(
        output.data(), output.size(),
        input.data(), input.size(),
        3 /* level */);
    if (ZSTD_isError(compressed)) {
        throw std::runtime_error(std::string("zstd compression failed: ") + ZSTD_getErrorName(compressed));
    }

    std::ofstream out(dst, std::ios::binary);
    out.write(reinterpret_cast<const char*>(output.data()), static_cast<std::streamsize>(compressed));
    if (!out) {
        throw std::runtime_error("failed to write zstd compressed file: " + dst.string());
    }
}

static std::string collect_runtime_fingerprint(int gpu_id) {
    std::ostringstream oss;

    int cuda_runtime = 0;
    if (cudaRuntimeGetVersion(&cuda_runtime) != cudaSuccess) {
        cuda_runtime = 0;
    }
    oss << "cuda_runtime=" << cuda_runtime;

#if defined(NV_TENSORRT_MAJOR) && defined(NV_TENSORRT_MINOR)
    oss << ";tensorrt=" << NV_TENSORRT_MAJOR << "." << NV_TENSORRT_MINOR;
#if defined(NV_TENSORRT_PATCH)
    oss << "." << NV_TENSORRT_PATCH;
#endif
#else
    oss << ";tensorrt=unknown";
#endif

#if defined(NVDS_VERSION_MAJOR)
    oss << ";deepstream=" << NVDS_VERSION_MAJOR;
#if defined(NVDS_VERSION_MINOR)
    oss << "." << NVDS_VERSION_MINOR;
#endif
#else
    oss << ";deepstream=9.0";
#endif

    nvmlReturn_t nvml_init = nvmlInit_v2();
    if (nvml_init == NVML_SUCCESS) {
        nvmlDevice_t device = nullptr;
        nvmlReturn_t dev_ret = nvmlDeviceGetHandleByIndex_v2(static_cast<unsigned int>(gpu_id), &device);
        if (dev_ret == NVML_SUCCESS && device != nullptr) {
            char uuid[NVML_DEVICE_UUID_V2_BUFFER_SIZE] = {0};
            if (nvmlDeviceGetUUID(device, uuid, sizeof(uuid)) == NVML_SUCCESS) {
                oss << ";gpu_uuid=" << uuid;
            }
        }
        nvmlShutdown();
    }

    char hostname[256] = {0};
    if (gethostname(hostname, sizeof(hostname)) == 0) {
        oss << ";hostname=" << hostname;
    }
    return oss.str();
}

struct RepresentativeCandidate {
    float score = -1.0f;
    int64_t pts_ns = -1;
    int32_t ordinal = -1;
    std::string observation_id;
    AlignedCropBuffer crop;
};

struct TrackAccumulator {
    std::string raw_track_key;
    int64_t first_pts_ns = -1;
    int64_t last_pts_ns = -1;
    int64_t observation_count = 0;
    int64_t eligible_observation_count = 0;
    std::vector<float> mean_embedding;
    bool has_template = false;
    RepresentativeCandidate representative;

    void update_template(const mergenvision::video::FaceDetection& face) {
        if (!face.recognition_eligible || face.embedding.size() != 512) {
            return;
        }
        if (mean_embedding.empty()) {
            mean_embedding.assign(face.embedding.begin(), face.embedding.end());
        } else {
            const float n = static_cast<float>(eligible_observation_count);
            for (size_t i = 0; i < 512; ++i) {
                mean_embedding[i] += (face.embedding[i] - mean_embedding[i]) / n;
            }
        }
        ++eligible_observation_count;
        has_template = true;
    }
};

class ArtifactState {
public:
    explicit ArtifactState(const std::string& output_dir,
                           const std::string& job_id,
                           const std::string& video_id,
                           int gpu_id,
                           const std::string& detector_engine,
                           const std::string& recognizer_engine)
        : output_dir_(output_dir),
          job_id_(job_id),
          video_id_(video_id),
          gpu_id_(gpu_id),
          detector_engine_(detector_engine),
          recognizer_engine_(recognizer_engine) {
        std::filesystem::create_directories(output_dir_ / "crops");
        obs_stream_.open(output_dir_ / "observations.pb", std::ios::binary);
        tmpl_stream_.open(output_dir_ / "track_templates.pb", std::ios::binary);
        if (!obs_stream_ || !tmpl_stream_) {
            throw std::runtime_error("failed to open artifact streams");
        }
    }

    ~ArtifactState() {
        if (obs_stream_.is_open()) obs_stream_.close();
        if (tmpl_stream_.is_open()) tmpl_stream_.close();
    }

    void write_observation_frame(const mergenvision::video::v1::VideoObservationFrame& frame) {
        google::protobuf::util::SerializeDelimitedToOstream(frame, &obs_stream_);
        ++observation_frame_count_;
    }

    void update_track(const std::string& raw_track_key,
                      const mergenvision::video::FaceDetection& face,
                      const AlignedCropBuffer& crop,
                      uint32_t display_width,
                      uint32_t display_height) {
        auto& acc = accumulators_[raw_track_key];
        acc.raw_track_key = raw_track_key;
        if (acc.first_pts_ns < 0 || face.frame.pts_ns < acc.first_pts_ns) {
            acc.first_pts_ns = face.frame.pts_ns;
        }
        if (face.frame.pts_ns > acc.last_pts_ns) {
            acc.last_pts_ns = face.frame.pts_ns;
        }
        ++acc.observation_count;
        acc.update_template(face);

        if (crop.size() != 3 * 112 * 112 || !face.recognition_eligible) {
            return;
        }

        const float w = std::max(1.0f, face.bbox.x2 - face.bbox.x1);
        const float h = std::max(1.0f, face.bbox.y2 - face.bbox.y1);
        const float face_area = w * h;
        const float image_area = static_cast<float>(display_width * display_height);
        float margin = 1.0f;
        if (face.bbox.x1 > 0 && face.bbox.y1 > 0 &&
            face.bbox.x2 < display_width && face.bbox.y2 < display_height) {
            margin = std::min({face.bbox.x1, face.bbox.y1,
                               display_width - face.bbox.x2,
                               display_height - face.bbox.y2});
        }
        const float score = face.quality_score * (face_area / image_area) * std::min(margin, 50.0f);

        RepresentativeCandidate cand;
        cand.score = score;
        cand.pts_ns = face.frame.pts_ns;
        cand.ordinal = static_cast<int32_t>(face.detection_ordinal);
        cand.observation_id = face.observation_id;
        cand.crop = crop;

        auto& rep = acc.representative;
        if (rep.observation_id.empty() ||
            cand.score > rep.score ||
            (cand.score == rep.score && cand.pts_ns < rep.pts_ns) ||
            (cand.score == rep.score && cand.pts_ns == rep.pts_ns && cand.ordinal < rep.ordinal)) {
            rep = std::move(cand);
        }
    }

    void finalize(const std::string& model_version,
                  const std::string& preprocess_version,
                  const std::string& config_version,
                  const std::string& input_video_path,
                  uint64_t wall_us) {
        (void)input_video_path;
        finalize_templates(model_version, preprocess_version, config_version);
        finalize_observations();
        write_crops();

        auto obs_pb = output_dir_ / "observations.pb";
        auto obs_zst = output_dir_ / "observations.pb.zst";
        auto tmpl_pb = output_dir_ / "track_templates.pb";
        auto tmpl_zst = output_dir_ / "track_templates.pb.zst";

        compress_file_zstd(obs_pb, obs_zst);
        delete_file_silent(obs_pb);
        compress_file_zstd(tmpl_pb, tmpl_zst);
        delete_file_silent(tmpl_pb);

        write_manifest(model_version, preprocess_version, config_version,
                       "", wall_us);
    }

    int64_t observation_frame_count() const { return observation_frame_count_; }
    int64_t raw_track_count() const { return static_cast<int64_t>(accumulators_.size()); }
    int64_t crop_count() const {
        int64_t n = 0;
        for (const auto& kv : accumulators_) {
            if (!kv.second.representative.crop.empty()) ++n;
        }
        return n;
    }

private:
    std::filesystem::path output_dir_;
    std::string job_id_;
    std::string video_id_;
    int gpu_id_ = 0;
    std::string detector_engine_;
    std::string recognizer_engine_;
    std::ofstream obs_stream_;
    std::ofstream tmpl_stream_;
    int64_t observation_frame_count_ = 0;
    std::unordered_map<std::string, TrackAccumulator> accumulators_;

    void finalize_observations() {
        mergenvision::video::v1::ObservationChunkFooter footer;
        footer.set_job_id(job_id_);
        footer.set_sequence_no(0);
        footer.set_frame_count(observation_frame_count_);
        google::protobuf::util::SerializeDelimitedToOstream(footer, &obs_stream_);
        obs_stream_.close();
    }

    void finalize_templates(const std::string& model_version,
                            const std::string& preprocess_version,
                            const std::string& config_version) {
        {
            mergenvision::video::v1::TrackTemplateBundle bundle;
            bundle.set_job_id(job_id_);
            bundle.set_video_id(video_id_);
            bundle.set_sequence_no(0);
            bundle.set_model_version(model_version);
            bundle.set_preprocess_version(preprocess_version);
            bundle.set_config_version(config_version);
            google::protobuf::util::SerializeDelimitedToOstream(bundle, &tmpl_stream_);
        }

        for (const auto& kv : accumulators_) {
            const auto& acc = kv.second;
            mergenvision::video::v1::RawTrackTemplate t;
            t.set_raw_track_key(acc.raw_track_key);
            t.set_first_pts_ns(acc.first_pts_ns);
            t.set_last_pts_ns(acc.last_pts_ns);
            t.set_observation_count(acc.observation_count);
            t.set_eligible_observation_count(acc.eligible_observation_count);
            if (acc.has_template && !acc.mean_embedding.empty()) {
                float norm = 0.0f;
                for (float v : acc.mean_embedding) norm += v * v;
                norm = std::sqrt(norm);
                const float inv_norm = norm > 1e-12f ? 1.0f / norm : 0.0f;
                for (float v : acc.mean_embedding) {
                    t.add_template_embedding(v * inv_norm);
                }
                t.set_template_quality(1.0f);  // TODO: replace with real quality estimate
            }
            if (!acc.representative.crop.empty()) {
                std::string rel = "crops/" + acc.raw_track_key + ".webp";
                t.set_representative_crop_relative_key(rel);
                t.set_representative_pts_ns(acc.representative.pts_ns);
                t.set_representative_ordinal(acc.representative.ordinal);
            } else {
                t.set_no_crop_reason("insufficient_representative_crop");
            }
            google::protobuf::util::SerializeDelimitedToOstream(t, &tmpl_stream_);
        }

        {
            mergenvision::video::v1::TrackTemplateFooter footer;
            footer.set_job_id(job_id_);
            footer.set_sequence_no(0);
            footer.set_template_count(static_cast<int64_t>(accumulators_.size()));
            google::protobuf::util::SerializeDelimitedToOstream(footer, &tmpl_stream_);
        }
        tmpl_stream_.close();
    }

    void write_crops() {
        for (const auto& kv : accumulators_) {
            const auto& acc = kv.second;
            if (acc.representative.crop.empty()) continue;
            auto path = output_dir_ / "crops" / (acc.raw_track_key + ".webp");
            std::vector<uint8_t> bytes = encode_webp(acc.representative.crop);
            std::ofstream f(path, std::ios::binary);
            f.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
        }
    }

    static std::vector<uint8_t> encode_webp(const AlignedCropBuffer& crop) {
        constexpr int k = 112;
        std::vector<uint8_t> rgb(k * k * 3);
        for (int y = 0; y < k; ++y) {
            for (int x = 0; x < k; ++x) {
                int plane = y * k + x;
                size_t idx = (y * k + x) * 3;
                auto denorm = [](float v) {
                    float p = v * 127.5f + 127.5f;
                    return static_cast<uint8_t>(std::clamp(p, 0.0f, 255.0f) + 0.5f);
                };
                rgb[idx + 0] = denorm(crop[plane + 0 * k * k]);
                rgb[idx + 1] = denorm(crop[plane + 1 * k * k]);
                rgb[idx + 2] = denorm(crop[plane + 2 * k * k]);
            }
        }
        uint8_t* out = nullptr;
        size_t size = WebPEncodeRGB(rgb.data(), k, k, k * 3, 90, &out);
        if (size == 0 || !out) {
            throw std::runtime_error("WebP encode failed");
        }
        std::vector<uint8_t> result(out, out + size);
        WebPFree(out);
        return result;
    }

    void write_manifest(const std::string& model_version,
                        const std::string& preprocess_version,
                        const std::string& config_version,
                        const std::string& input_sha256,
                        uint64_t wall_us) {
        (void)input_sha256;
        auto obs_path = output_dir_ / "observations.pb.zst";
        auto tmpl_path = output_dir_ / "track_templates.pb.zst";
        int64_t crop_n = crop_count();
        const std::string runtime_fingerprint = collect_runtime_fingerprint(gpu_id_);

        std::ostringstream oss;
        oss << "{\n";
        oss << "  \"schema_versions\": {\n";
        oss << "    \"observation\": \"mergenvision.video.v1.VideoObservationFrame\",\n";
        oss << "    \"template\": \"mergenvision.video.v1.TrackTemplateBundle\",\n";
        oss << "    \"manifest\": \"1\"\n";
        oss << "  },\n";
        oss << "  \"job_id\": \"" << escape_json(job_id_) << "\",\n";
        oss << "  \"video_id\": \"" << escape_json(video_id_) << "\",\n";
        oss << "  \"model_version\": \"" << escape_json(model_version) << "\",\n";
        oss << "  \"preprocess_version\": \"" << escape_json(preprocess_version) << "\",\n";
        oss << "  \"config_version\": \"" << escape_json(config_version) << "\",\n";
        oss << "  \"observation_frame_count\": " << observation_frame_count_ << ",\n";
        oss << "  \"raw_track_count\": " << accumulators_.size() << ",\n";
        oss << "  \"template_count\": " << accumulators_.size() << ",\n";
        oss << "  \"crop_count\": " << crop_n << ",\n";
        oss << "  \"runtime_fingerprint\": \"" << escape_json(runtime_fingerprint) << "\",\n";
        oss << "  \"artifacts\": {\n";
        oss << "    \"observations.pb.zst\": {\"size\": " << std::filesystem::file_size(obs_path) << "},\n";
        oss << "    \"track_templates.pb.zst\": {\"size\": " << std::filesystem::file_size(tmpl_path) << "}";

        std::vector<std::string> crop_entries;
        for (const auto& kv : accumulators_) {
            const auto& acc = kv.second;
            if (acc.representative.crop.empty()) continue;
            auto crop_path = output_dir_ / "crops" / (acc.raw_track_key + ".webp");
            std::error_code ec;
            if (!std::filesystem::exists(crop_path, ec)) continue;
            std::string rel = "crops/" + acc.raw_track_key + ".webp";
            std::ostringstream entry;
            entry << "    \"" << escape_json(rel) << "\": {\"size\": "
                  << std::filesystem::file_size(crop_path) << "}";
            crop_entries.push_back(entry.str());
        }
        for (const auto& entry : crop_entries) {
            oss << ",\n" << entry;
        }
        oss << "\n  },\n";
        oss << "  \"wall_us\": " << wall_us << "\n";
        oss << "}\n";

        std::ofstream f(output_dir_ / "manifest.json");
        f << oss.str();
    }

    static std::string escape_json(const std::string& s) {
        std::string out;
        out.reserve(s.size());
        for (char c : s) {
            switch (c) {
                case '"': out += "\\\""; break;
                case '\\': out += "\\\\"; break;
                case '\b': out += "\\b"; break;
                case '\f': out += "\\f"; break;
                case '\n': out += "\\n"; break;
                case '\r': out += "\\r"; break;
                case '\t': out += "\\t"; break;
                default: out += c; break;
            }
        }
        return out;
    }
};

#endif  // MV_VIDEO_WORKER

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

#ifdef MV_VIDEO_WORKER
// Build a UI-safe observation frame protobuf. For this vertical slice we still
// include per-detection embeddings because downstream Python reconciliation
// services consume them; they will be moved to the template artifact in a
// follow-up refactor of those services.
static mergenvision::video::v1::VideoObservationFrame build_proto_frame(
    const std::string& job_id,
    const std::string& video_id,
    const mergenvision::video::FrameDetections& fd) {
    using FD = mergenvision::video::FaceDetection;
    mergenvision::video::v1::VideoObservationFrame proto;
    proto.set_job_id(job_id);
    proto.set_video_id(video_id);
    proto.set_stream_index(0);
    proto.set_frame_index(static_cast<int64_t>(fd.frame.presentation_index));
    proto.set_source_pts(fd.frame.pts_ns);
    proto.set_pts_ns(fd.frame.pts_ns);
    proto.set_time_base_num(1);
    proto.set_time_base_den(1'000'000'000);
    proto.set_display_width(static_cast<int32_t>(fd.frame.display_width));
    proto.set_display_height(static_cast<int32_t>(fd.frame.display_height));
    proto.set_rotation(fd.frame.rotation_degrees);

    for (const FD& face : fd.detections) {
        auto* pd = proto.add_detections();
        pd->set_detection_id(face.observation_id);
        pd->set_ordinal(static_cast<int32_t>(face.detection_ordinal));
        pd->set_x(static_cast<int32_t>(face.bbox.x1));
        pd->set_y(static_cast<int32_t>(face.bbox.y1));
        pd->set_width(static_cast<int32_t>(face.bbox.x2 - face.bbox.x1));
        pd->set_height(static_cast<int32_t>(face.bbox.y2 - face.bbox.y1));
        for (float v : face.landmarks) pd->add_landmarks(v);
        pd->set_detector_score(face.detector_score);
        pd->set_quality_score(face.quality_score);
        pd->set_tracking_eligible(face.tracking_eligible);
        pd->set_recognition_eligible(face.recognition_eligible);
        pd->set_rejection_code(face.rejection_code);
        if (face.recognition_eligible && face.embedding.size() == 512) {
            for (float v : face.embedding) pd->add_embedding(v);
        }
        pd->set_model_version(face.model_version);
        pd->set_preprocess_version(face.preprocess_version);
        pd->set_raw_track_key(face.raw_track_key);
    }
    return proto;
}
#endif

static void process_complete_inference_batch(
    const mergenvision::video::InferenceFrameBatch& batch,
    ProbeState* ps,
    SmokeStats* stats) {
    auto frame_detections = ps->pipeline->infer_detector_batch(batch);
#ifdef MV_VIDEO_WORKER
    auto aligned_crops = ps->pipeline->take_recognition_crops();
#endif
    auto m = ps->pipeline->metrics();

    // Verify full observation output: every detection must now carry a valid,
    // L2-normalized 512-D embedding from the RGBA warp-align + GlintR100 path.
    {
        std::lock_guard<std::mutex> lock(stats->mtx);
        for (const auto& fd : frame_detections) {
            for (const auto& face : fd.detections) {
                if (!face.recognition_eligible || face.embedding.empty()) continue;
                ++stats->total_embeddings;
                const size_t expected = 512;
                if (face.embedding.size() != expected) {
                    ++stats->embedding_dim_errors;
                    continue;
                }
                double norm_sq = 0.0;
                bool finite = true;
                for (float v : face.embedding) {
                    if (!std::isfinite(v)) finite = false;
                    norm_sq += static_cast<double>(v) * static_cast<double>(v);
                }
                if (!finite) {
                    ++stats->embedding_finite_errors;
                    continue;
                }
                double norm = std::sqrt(norm_sq);
                if (stats->total_embeddings == 1) {
                    stats->embedding_norm_min = norm;
                    stats->embedding_norm_max = norm;
                } else {
                    stats->embedding_norm_min = std::min(stats->embedding_norm_min, norm);
                    stats->embedding_norm_max = std::max(stats->embedding_norm_max, norm);
                }
            }
        }
    }

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

#ifdef MV_VIDEO_WORKER
        if (ps->artifact) {
            // Tag detections with raw-track correlation before building the
            // observation artifact.  |tracked| owns copies of the FaceDetection
            // structs, so we write the key back into |fd.detections| by
            // observation_id before serializing the frame.
            std::unordered_map<std::string, std::string> obs_to_track;
            for (const auto& tracked_det : tracked) {
                obs_to_track[tracked_det.detection.observation_id] = tracked_det.local_track_key;
            }
            for (auto& det : fd.detections) {
                auto it = obs_to_track.find(det.observation_id);
                if (it != obs_to_track.end()) {
                    det.raw_track_key = it->second;
                }
            }
            ps->artifact->write_observation_frame(
                build_proto_frame(ps->opts->job_id, ps->opts->video_id, fd));

            for (const auto& tracked_det : tracked) {
                const mergenvision::video::FaceDetection& face = tracked_det.detection;
                auto it = aligned_crops.find(face.observation_id);
                const AlignedCropBuffer* crop = (it != aligned_crops.end()) ? &it->second : nullptr;
                AlignedCropBuffer empty_crop;
                ps->artifact->update_track(
                    tracked_det.local_track_key,
                    face,
                    crop ? *crop : empty_crop,
                    fd.frame.display_width,
                    fd.frame.display_height);
            }
        }
#endif
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
        stats->pipeline_recognition_us += m.recognition_us;
        stats->pipeline_calls += m.total_calls;
    }
}

static void processing_thread_loop(WorkQueue* queue,
                                   ProbeState* ps,
                                   SmokeStats* stats) {
    auto* assembler = ps->assembler;
    try {
        while (auto maybe_frames = queue->pop()) {
            auto complete_batches = assembler->push(std::move(maybe_frames.value()));
            for (auto& inf_batch : complete_batches) {
                process_complete_inference_batch(inf_batch, ps, stats);
            }
        }
        // End of stream: flush any partial inference batch.
        auto final_batch = assembler->flush_eos();
        if (final_batch) {
            process_complete_inference_batch(final_batch.value(), ps, stats);
            {
                std::lock_guard<std::mutex> lock(stats->mtx);
                ++stats->partial_batches;
            }
        }
    } catch (const std::exception& e) {
        queue->set_error(std::string("processing thread error: ") + e.what());
    }
}

static GstPadProbeReturn on_mux_src_buffer(GstPad* pad, GstPadProbeInfo* info, gpointer user_data) {
    auto* ps = static_cast<ProbeState*>(user_data);
    auto* stats = ps->stats;
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

    // Hand off retained frames to the background processing thread. The pad
    // probe must return immediately so GStreamer decode is not throttled.
    ps->queue->push(std::move(mux_frames));

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
#ifdef MV_VIDEO_WORKER
        } else if (arg == "--output" && i + 1 < argc) {
            opts->output_dir = argv[++i];
        } else if (arg == "--job-id" && i + 1 < argc) {
            opts->job_id = argv[++i];
        } else if (arg == "--video-id" && i + 1 < argc) {
            opts->video_id = argv[++i];
        } else if (arg == "--model-profile" && i + 1 < argc) {
            opts->model_profile = argv[++i];
        } else if (arg == "--detector-engine" && i + 1 < argc) {
            opts->detector_engine = argv[++i];
        } else if (arg == "--recognizer-engine" && i + 1 < argc) {
            opts->recognizer_engine = argv[++i];
#endif
        } else if (arg == "--help" || arg == "-h") {
            std::fprintf(stdout,
                "usage: %s --input <video_path> [--max-frames N | --all-frames]\n"
                "       [--gpu-id ID] [--detector-batch-size N] [--recognizer-batch-size N]\n"
#ifdef MV_VIDEO_WORKER
                "       [--output <dir>] [--job-id <uuid>] [--video-id <uuid>]\n"
#endif
                ,
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
#ifdef MV_VIDEO_WORKER
    if (opts->output_dir.empty() || opts->job_id.empty() || opts->video_id.empty() ||
        opts->detector_engine.empty() || opts->recognizer_engine.empty()) {
        std::fprintf(stderr,
            "error: --output, --job-id, --video-id, --detector-engine, --recognizer-engine are required\n");
        return false;
    }
#endif
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

#ifdef MV_VIDEO_WORKER
    const std::string retina_engine_path = opts.detector_engine;
    const std::string glint_engine_path = opts.recognizer_engine;
#else
    const std::string retina_engine_path =
        "backend/artifacts/engines/deepstream9/retinaface_r50_dynamic.bs1.opt8.max64.fp16.trt1014.engine";
    const std::string glint_engine_path =
        "backend/artifacts/engines/deepstream9/glintr100.bs1.opt8.max64.fp16.trt1014.engine";
#endif

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
    WorkQueue queue;
    ps.opts = &opts;
    ps.stats = &stats;
    ps.queue = &queue;
    ps.assembler = &assembler;
    ps.pipeline = &face_pipeline;
    ps.tracker = &tracker;

#ifdef MV_VIDEO_WORKER
    std::optional<ArtifactState> artifact_state;
    std::filesystem::path worker_tmp_output;
    if (!opts.output_dir.empty()) {
        worker_tmp_output = opts.output_dir;
        worker_tmp_output += ".tmp";
        std::filesystem::remove_all(worker_tmp_output);
        artifact_state.emplace(
            worker_tmp_output.string(),
            opts.job_id,
            opts.video_id,
            opts.gpu_id,
            opts.detector_engine,
            opts.recognizer_engine);
        ps.artifact = &*artifact_state;
    }
#endif

    std::thread processing_thread(processing_thread_loop, &queue, &ps, &stats);

    PipelineContext ctx;
    ctx.pipeline = gst_pipeline_new("real-batching-smoke");
    ctx.demux = make_element_checked("qtdemux", "demux", &ctx);
    ctx.parser = make_element_checked("h264parse", "parser", &ctx);
    ctx.decoder = make_element_checked("nvv4l2decoder", "decoder", &ctx);
    ctx.mux = make_element_checked("nvstreammux", "mux", &ctx);
    GstElement* streamdemux = make_element_checked("nvstreamdemux", "streamdemux", &ctx);
    GstElement* queue_element = make_element_checked("queue", "queue", &ctx);
    GstElement* sink = make_element_checked("fakesink", "sink", &ctx);

    if (!ctx.pipeline || !ctx.demux || !ctx.parser || !ctx.decoder || !ctx.mux ||
        !streamdemux || !queue_element || !sink) {
        return 1;
    }

    gst_bin_add_many(GST_BIN(ctx.pipeline),
        ctx.demux, ctx.parser, ctx.decoder, ctx.mux,
        streamdemux, queue_element, sink, nullptr);

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
        "width", 1920,
        "height", 1080,
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
    if (!link_static(queue_element, sink, &ctx)) return 1;

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

    GstPad* streamdemux_src = gst_element_request_pad_simple(streamdemux, "src_0");
    GstPad* queue_sink = gst_element_get_static_pad(queue_element, "sink");
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

    // Signal the processing thread to drain and flush any partial batch.
    queue.set_eos();
    processing_thread.join();
    {
        std::lock_guard<std::mutex> lock(stats.mtx);
        if (queue.error && !stats.error) {
            stats.error = true;
            stats.error_message = queue.error_message;
        }
    }

    gst_element_set_state(ctx.pipeline, GST_STATE_NULL);
    gst_object_unref(bus);
    if (ctx.mux_sink_pad) {
        gst_element_release_request_pad(ctx.mux, ctx.mux_sink_pad);
        gst_object_unref(ctx.mux_sink_pad);
    }
    gst_object_unref(ctx.pipeline);

#ifdef MV_VIDEO_WORKER
    if (artifact_state) {
        const uint64_t wall_us = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - t_start).count());
        artifact_state->finalize(
            opts.model_profile.empty() ? "retinaface_r50_glintr100_v1" : opts.model_profile,
            "cuda_five_point_align",
            "1",
            opts.video_path,
            wall_us);
        std::filesystem::remove_all(opts.output_dir);
        std::filesystem::rename(worker_tmp_output, opts.output_dir);
    }
#endif

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
    uint64_t final_pipeline_recognition_us = 0;
    uint64_t final_pipeline_calls = 0;
    int final_embeddings = 0;
    int final_embedding_dim_errors = 0;
    int final_embedding_finite_errors = 0;
    double final_embedding_norm_min = 0.0;
    double final_embedding_norm_max = 0.0;

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
        final_pipeline_recognition_us = stats.pipeline_recognition_us;
        final_pipeline_calls = stats.pipeline_calls;
        final_embeddings = stats.total_embeddings;
        final_embedding_dim_errors = stats.embedding_dim_errors;
        final_embedding_finite_errors = stats.embedding_finite_errors;
        final_embedding_norm_min = stats.embedding_norm_min;
        final_embedding_norm_max = stats.embedding_norm_max;
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
    g_print("embeddings: %d\n", final_embeddings);
    g_print("embedding norm: %.5f .. %.5f\n", final_embedding_norm_min, final_embedding_norm_max);
    g_print("NVMM device memory: %s\n", final_got_nvm ? "yes" : "no");
    g_print("PTS range: %" G_GINT64_FORMAT " .. %" G_GINT64_FORMAT " ns\n",
            final_first_pts, final_last_pts);
    g_print("wall time: %.2f ms\n", elapsed_ms);
    g_print("pipeline fps: %.2f\n", fps);
    if (final_pipeline_calls > 0) {
        auto avg = [&](uint64_t us) { return static_cast<double>(us) / static_cast<double>(final_pipeline_calls); };
        g_print("per-call avg (us): preprocess=%.0f enqueue=%.0f postproc=%.0f mapping=%.0f recognition=%.0f tracker=%.0f\n",
                avg(final_pipeline_preprocess_us),
                avg(final_pipeline_engine_us),
                avg(final_pipeline_postproc_us),
                avg(final_pipeline_mapping_us),
                avg(final_pipeline_recognition_us),
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
    if (final_embeddings == 0) {
        std::fprintf(stderr, "no embeddings produced\n");
        pass = false;
    }
    if (final_embedding_dim_errors > 0) {
        std::fprintf(stderr, "embedding dimension errors: %d\n", final_embedding_dim_errors);
        pass = false;
    }
    if (final_embedding_finite_errors > 0) {
        std::fprintf(stderr, "embedding non-finite errors: %d\n", final_embedding_finite_errors);
        pass = false;
    }
    if (final_embeddings > 0 &&
        (final_embedding_norm_min < 0.98 || final_embedding_norm_max > 1.02)) {
        std::fprintf(stderr, "embedding L2 norm out of range: %.5f .. %.5f\n",
                     final_embedding_norm_min, final_embedding_norm_max);
        pass = false;
    }

    if (pass) {
        g_print("real batching smoke PASSED\n");
        return 0;
    }
    std::fprintf(stderr, "real batching smoke FAILED\n");
    return 1;
}

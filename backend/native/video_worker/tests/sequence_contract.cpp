// M5.1 — Sequential frame/batch contract unit tests.
//
// These are CPU-only metadata tests. They prove that frame identity, temporal
// batch assembly, detector/recognition mapping and tracker chronology behave
// deterministically and correctly across detector batch boundaries.

#include "mv/video/batch_assembler.hpp"
#include "mv/video/buffer_owner.hpp"
#include "mv/video/detection_mapper.hpp"
#include "mv/video/frame_identity.hpp"
#include "mv/video/recognition_mapper.hpp"
#include "mv/video/tracker_adapter.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace mv = mergenvision::video;

namespace {

int g_failures = 0;

#define CHECK(cond, msg)                                                \
    do {                                                                \
        if (!(cond)) {                                                  \
            std::cerr << "  CHECK FAILED: " << msg << " at line " << __LINE__ << "\n"; \
            ++g_failures;                                               \
            return false;                                               \
        }                                                               \
    } while (0)

mv::FrameEnvelope make_frame(uint64_t presentation_index,
                             int64_t pts_ns,
                             uint64_t mux_batch_sequence,
                             uint32_t position_in_mux_batch,
                             uint64_t nvds_frame_num,
                             int retained_id) {
    mv::FrameEnvelope env;
    env.presentation_index = presentation_index;
    env.decoded_sequence = presentation_index;
    env.sampled_sequence = presentation_index;
    env.mux_batch_sequence = mux_batch_sequence;
    env.position_in_mux_batch = position_in_mux_batch;
    env.source_id = 0;
    env.pad_index = 0;
    env.nvds_frame_num = nvds_frame_num;
    env.pts_ns = pts_ns;
    env.duration_ns = 0;
    env.pts_derived = false;
    env.coded_width = 1920;
    env.coded_height = 1080;
    env.display_width = 1920;
    env.display_height = 1080;
    env.rotation_degrees = 0;
    env.device_view.width = 1920;
    env.device_view.height = 1080;
    env.owner = mv::make_fake_retained_handle(retained_id);
    return env;
}

mv::FaceDetection make_detection(uint64_t presentation_index,
                                 uint32_t ordinal,
                                 float x1,
                                 float y1,
                                 float x2,
                                 float y2,
                                 float score = 0.9f) {
    mv::FaceDetection det;
    det.frame.presentation_index = presentation_index;
    det.detection_ordinal = ordinal;
    mv::BBox bbox;
    bbox.x1 = x1;
    bbox.y1 = y1;
    bbox.x2 = x2;
    bbox.y2 = y2;
    det.bbox = bbox;
    det.detector_score = score;
    det.recognition_eligible = false;
    det.tracking_eligible = true;
    det.landmarks.fill(0.0f);
    return det;
}

}  // namespace

// ---------------------------------------------------------------------------
// A. Sequential assembler: F0..F17 with max_batch=8 yields exactly three
// inference batches: [0..7], [8..15], [16..17].
// ---------------------------------------------------------------------------
static bool test_sequential_assembler() {
    mv::TemporalFrameBatchAssembler assembler(8, false);
    std::vector<mv::FrameEnvelope> frames;
    for (uint64_t i = 0; i < 18; ++i) {
        frames.push_back(make_frame(i, static_cast<int64_t>(i) * 40'000'000, 0,
                                    static_cast<uint32_t>(i), i, static_cast<int>(i)));
    }
    auto batches = assembler.push(std::move(frames));
    auto last = assembler.flush_eos();

    CHECK(batches.size() == 2, "expected 2 complete batches before flush");
    CHECK(batches[0].frames.size() == 8, "batch 0 size");
    CHECK(batches[1].frames.size() == 8, "batch 1 size");
    CHECK(last.has_value(), "expected eos flush");
    CHECK(last->frames.size() == 2, "eos batch size");

    for (size_t b = 0; b < batches.size(); ++b) {
        CHECK(batches[b].batch_sequence == b, "batch sequence");
        for (size_t i = 0; i < batches[b].frames.size(); ++i) {
            const auto& f = batches[b].frames[i];
            CHECK(f.inference_batch_sequence == b, "inference_batch_sequence");
            CHECK(f.position_in_inference_batch == i, "position_in_inference_batch");
            CHECK(f.presentation_index == b * 8 + i, "presentation_index");
            CHECK(f.pts_ns == static_cast<int64_t>(f.presentation_index) * 40'000'000,
                  "pts_ns");
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// B. Irregular mux buffers: chunks [0,1,2], [3], [4,5,6,7,8], [9] must produce
// inference batches [0..7] and [8,9].
// ---------------------------------------------------------------------------
static bool test_irregular_mux_buffers() {
    mv::TemporalFrameBatchAssembler assembler(8, false);
    std::vector<std::vector<uint64_t>> chunks = {
        {0, 1, 2}, {3}, {4, 5, 6, 7, 8}, {9}};

    std::vector<mv::InferenceFrameBatch> emitted;
    uint64_t mux_seq = 0;
    for (const auto& chunk : chunks) {
        std::vector<mv::FrameEnvelope> frames;
        for (size_t i = 0; i < chunk.size(); ++i) {
            auto idx = chunk[i];
            frames.push_back(make_frame(idx, static_cast<int64_t>(idx) * 40'000'000,
                                        mux_seq, static_cast<uint32_t>(i), idx,
                                        static_cast<int>(idx)));
        }
        ++mux_seq;
        auto out = assembler.push(std::move(frames));
        emitted.insert(emitted.end(), std::make_move_iterator(out.begin()),
                       std::make_move_iterator(out.end()));
    }
    auto last = assembler.flush_eos();
    if (last.has_value()) emitted.push_back(std::move(*last));

    CHECK(emitted.size() == 2, "expected 2 inference batches");
    CHECK(emitted[0].frames.size() == 8, "batch 0 must be F0..F7");
    CHECK(emitted[1].frames.size() == 2, "batch 1 must be F8,F9");
    for (size_t i = 0; i < 8; ++i) {
        CHECK(emitted[0].frames[i].presentation_index == i, "batch 0 presentation index");
    }
    CHECK(emitted[1].frames[0].presentation_index == 8, "batch 1 first");
    CHECK(emitted[1].frames[1].presentation_index == 9, "batch 1 second");
    return true;
}

// ---------------------------------------------------------------------------
// C. PTS ordering inside a single mux buffer: input order [F2,F0,F1] (PTS
// 200ms, 0ms, 100ms) must be canonicalized to [F0,F1,F2].
// ---------------------------------------------------------------------------
static bool test_pts_ordering() {
    mv::TemporalFrameBatchAssembler assembler(8, false);
    std::vector<mv::FrameEnvelope> frames;
    frames.push_back(make_frame(2, 200'000'000, 0, 2, 2, 2));
    frames.push_back(make_frame(0, 0, 0, 0, 0, 0));
    frames.push_back(make_frame(1, 100'000'000, 0, 1, 1, 1));

    auto batches = assembler.push(std::move(frames));
    auto last = assembler.flush_eos();

    CHECK(batches.empty(), "no full batch yet");
    CHECK(last.has_value(), "eos flush expected");
    CHECK(last->frames.size() == 3, "three frames after reorder");
    CHECK(last->frames[0].presentation_index == 0, "first is F0");
    CHECK(last->frames[1].presentation_index == 1, "second is F1");
    CHECK(last->frames[2].presentation_index == 2, "third is F2");
    CHECK(last->frames[0].pts_ns == 0, "F0 pts");
    CHECK(last->frames[1].pts_ns == 100'000'000, "F1 pts");
    CHECK(last->frames[2].pts_ns == 200'000'000, "F2 pts");
    return true;
}

// ---------------------------------------------------------------------------
// D. PTS regression raises VIDEO_PRESENTATION_ORDER_VIOLATION.
// ---------------------------------------------------------------------------
static bool test_pts_regression() {
    mv::TemporalFrameBatchAssembler assembler(8, false);
    std::vector<mv::FrameEnvelope> first;
    first.push_back(make_frame(0, 100'000'000, 0, 0, 0, 0));
    assembler.push(std::move(first));

    std::vector<mv::FrameEnvelope> second;
    second.push_back(make_frame(1, 80'000'000, 1, 0, 1, 1));
    bool threw = false;
    try {
        assembler.push(std::move(second));
    } catch (const mv::PresentationOrderViolation& e) {
        threw = true;
        std::string what = e.what();
        CHECK(what.find("VIDEO_PRESENTATION_ORDER_VIOLATION") != std::string::npos,
              "exception must contain code");
    }
    CHECK(threw, "expected PresentationOrderViolation");
    return true;
}

// ---------------------------------------------------------------------------
// E. Sampling: decoded F0..F9, every_n=3 → processed presentation indices
// [0,3,6,9] with sampled_sequence 0..3.
// ---------------------------------------------------------------------------
static bool test_sampling() {
    // Sampling filter applied before the assembler.
    std::vector<mv::FrameEnvelope> sampled;
    int sampled_seq = 0;
    for (uint64_t i = 0; i < 10; ++i) {
        if (i % 3 == 0) {
            auto env = make_frame(i, static_cast<int64_t>(i) * 40'000'000, 0,
                                  static_cast<uint32_t>(sampled_seq), i,
                                  static_cast<int>(i));
            env.sampled_sequence = static_cast<uint64_t>(sampled_seq);
            sampled.push_back(env);
            ++sampled_seq;
        }
    }

    mv::TemporalFrameBatchAssembler assembler(8, true);
    auto batches = assembler.push(std::move(sampled));
    auto last = assembler.flush_eos();

    CHECK(batches.empty(), "not enough frames for full batch");
    CHECK(last.has_value(), "eos flush");
    CHECK(last->frames.size() == 4, "four sampled frames");

    uint64_t expected_pidx[] = {0, 3, 6, 9};
    for (size_t i = 0; i < 4; ++i) {
        CHECK(last->frames[i].presentation_index == expected_pidx[i],
              "presentation_index matches every_n");
        CHECK(last->frames[i].sampled_sequence == i, "sampled_sequence");
    }
    return true;
}

// ---------------------------------------------------------------------------
// F. Owner lifetime: retained handles are not released before the inference
// batch is consumed and are released on EOS/error/cancel.
// ---------------------------------------------------------------------------
static bool test_owner_lifetime() {
    // Observe owners through weak_ptrs so we can prove the handle is destroyed
    // exactly when the last retained reference (the pending/emitted batch) is
    // gone. The production contract is: retained GstBuffer refs are released
    // once the inference batch no longer needs the device view.
    std::vector<std::weak_ptr<mv::RetainedBufferHandle>> weak_owners;
    {
        mv::TemporalFrameBatchAssembler assembler(8, false);
        std::vector<mv::FrameEnvelope> frames;
        for (int i = 0; i < 4; ++i) {
            frames.push_back(make_frame(i, static_cast<int64_t>(i) * 40'000'000, 0,
                                        static_cast<uint32_t>(i), i, i));
            weak_owners.push_back(frames.back().owner);
        }
        auto batches = assembler.push(std::move(frames));
        CHECK(batches.empty(), "not enough for full batch");
        CHECK(assembler.pending_count() == 4, "four pending frames");

        for (const auto& weak : weak_owners) {
            auto owner = weak.lock();
            CHECK(owner && !owner->is_released(), "pending owner not released");
        }

        auto last = assembler.flush_eos();
        CHECK(last.has_value(), "eos emits partial batch");
        for (const auto& weak : weak_owners) {
            auto owner = weak.lock();
            CHECK(owner && !owner->is_released(), "owner still retained after eos emits batch");
        }
    }

    // After the batch and assembler are destroyed, no references remain and
    // the fake handle destructor releases the handle.
    for (const auto& weak : weak_owners) {
        CHECK(weak.expired(), "owner handle destroyed after batch destruction");
    }

    // Cancel path: cancel must drop pending frames and release their owners.
    {
        mv::TemporalFrameBatchAssembler assembler(8, false);
        std::vector<mv::FrameEnvelope> frames;
        frames.push_back(make_frame(0, 0, 0, 0, 0, 100));
        std::weak_ptr<mv::RetainedBufferHandle> weak = frames.back().owner;
        assembler.push(std::move(frames));
        assembler.cancel();
        CHECK(weak.expired(), "owner handle destroyed after cancel");
    }

    return true;
}

// ---------------------------------------------------------------------------
// G. Detector mapping: even if raw output order is shuffled by position, each
// detection maps back to the frame it belongs to.
// ---------------------------------------------------------------------------
static bool test_detector_mapping() {
    mv::InferenceFrameBatch batch;
    batch.batch_sequence = 0;
    for (uint64_t i = 0; i < 4; ++i) {
        batch.frames.push_back(make_frame(i, static_cast<int64_t>(i) * 40'000'000, 0,
                                          static_cast<uint32_t>(i), i,
                                          static_cast<int>(i) + 1000));
    }

    // Raw detections are deliberately out of order.
    std::vector<mv::RawDetection> raw = {
        {2, {10.0f, 10.0f, 20.0f, 20.0f}, 0.95f, {}, false},
        {0, {0.0f, 0.0f, 5.0f, 5.0f}, 0.90f, {}, false},
        {3, {30.0f, 30.0f, 40.0f, 40.0f}, 0.88f, {}, false},
        {1, {5.0f, 5.0f, 10.0f, 10.0f}, 0.85f, {}, false},
    };

    auto mapped = mv::map_detector_output_to_frames(batch, raw, "job_42");
    CHECK(mapped.size() == 4, "one FrameDetections per input frame");

    CHECK(mapped[0].frame.presentation_index == 0, "frame 0 identity");
    CHECK(mapped[0].detections.size() == 1, "frame 0 has one detection");
    CHECK(mapped[0].detections[0].bbox.x1 == 0.0f, "frame 0 bbox");

    CHECK(mapped[1].frame.presentation_index == 1, "frame 1 identity");
    CHECK(mapped[1].detections[0].bbox.x1 == 5.0f, "frame 1 bbox");

    CHECK(mapped[2].detections[0].bbox.x1 == 10.0f, "frame 2 bbox");
    CHECK(mapped[3].detections[0].bbox.x1 == 30.0f, "frame 3 bbox");

    // Observation id format invariant.
    CHECK(mapped[2].detections[0].observation_id == "job_42:2:0",
          "observation_id format");
    return true;
}

// ---------------------------------------------------------------------------
// H. Recognition mapping: 33 crops chunk into 32 + 1 and embeddings reattach to
// the exact (presentation_index, detection_ordinal).
// ---------------------------------------------------------------------------
static bool test_recognition_mapping() {
    std::vector<mv::RecognitionCropRef> crops;
    for (int i = 0; i < 33; ++i) {
        mv::RecognitionCropRef ref;
        ref.presentation_index = static_cast<uint64_t>(i / 2);
        ref.detection_ordinal = static_cast<uint32_t>(i % 2);
        ref.detector_batch_position = static_cast<uint32_t>(i % 8);
        ref.crop = {nullptr, 112, 112, 112};
        crops.push_back(ref);
    }

    auto chunks = mv::chunk_recognition_crops(crops, 32);
    CHECK(chunks.size() == 2, "33 crops → two chunks");
    CHECK(chunks[0].size() == 32, "first chunk full");
    CHECK(chunks[1].size() == 1, "second chunk remainder");

    std::vector<std::vector<float>> chunk_embeddings;
    for (size_t c = 0; c < chunks.size(); ++c) {
        std::vector<float> embs;
        embs.resize(chunks[c].size() * 512);
        for (size_t i = 0; i < chunks[c].size(); ++i) {
            // Make each embedding unique by its canonical identity hash so
            // reattachment is unambiguous.
            float base = static_cast<float>(chunks[c][i].presentation_index * 100 +
                                            chunks[c][i].detection_ordinal);
            for (size_t j = 0; j < 512; ++j) {
                embs[i * 512 + j] = base + static_cast<float>(j) / 1000.0f;
            }
        }
        chunk_embeddings.push_back(std::move(embs));
    }

    auto results = mv::map_recognition_embeddings(chunks, chunk_embeddings);
    CHECK(results.size() == 33, "33 embeddings returned");

    for (const auto& res : results) {
        uint64_t expected_pidx = res.embedding[0] / 100.0f;
        uint32_t expected_ordinal =
            static_cast<uint32_t>(res.embedding[0] - static_cast<float>(expected_pidx) * 100.0f);
        CHECK(res.presentation_index == expected_pidx, "embedding reattached to correct frame");
        CHECK(res.detection_ordinal == expected_ordinal,
              "embedding reattached to correct detection");
    }
    return true;
}

// ---------------------------------------------------------------------------
// I. Track batch boundary: a single object visible in frames 6..10 receives the
// same local_track_key, even though frame 7 is the last frame of batch 0 and
// frame 8 is the first frame of batch 1 (max_batch=8).
// ---------------------------------------------------------------------------
static bool test_track_batch_boundary() {
    mv::TemporalFrameBatchAssembler assembler(8, false);
    std::vector<mv::FaceDetection> all_detections;
    for (uint64_t i = 6; i <= 10; ++i) {
        all_detections.push_back(make_detection(i, 0, 100.0f, 100.0f, 120.0f, 120.0f, 0.95f));
    }

    // Feed frames one at a time to simulate sequential tracker updates after
    // detector batching. The tracker instance must survive across batches.
    mv::NaiveTracker tracker;
    std::string key;
    for (uint64_t i = 6; i <= 10; ++i) {
        std::vector<mv::FaceDetection> frame_dets;
        frame_dets.push_back(all_detections[static_cast<size_t>(i - 6)]);
        auto tracked = tracker.update(i, static_cast<int64_t>(i) * 40'000'000, frame_dets);
        CHECK(tracked.size() == 1, "one tracked detection per frame");
        if (i == 6) {
            key = tracked[0].local_track_key;
            CHECK(key == "RT000001", "first track key");
        } else {
            CHECK(tracked[0].local_track_key == key, "same track across batch boundary");
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// J. Chunk invariance: identical synthetic metadata run with detector batch
// sizes 1, 4 and 8 produces the same logical track assignments.
// ---------------------------------------------------------------------------
static bool test_chunk_invariance() {
    // A person moving linearly: frames 0..11, bbox shifts by (2,2) per frame.
    auto run_with_batch_size = [](size_t batch_size) -> std::vector<std::string> {
        std::vector<mv::FaceDetection> detections;
        for (uint64_t i = 0; i < 12; ++i) {
            float off = static_cast<float>(i) * 2.0f;
            detections.push_back(
                make_detection(i, 0, off, off, off + 20.0f, off + 20.0f, 0.95f));
        }

        mv::TemporalFrameBatchAssembler assembler(batch_size, false);
        std::vector<mv::FrameEnvelope> frames;
        for (uint64_t i = 0; i < 12; ++i) {
            frames.push_back(
                make_frame(i, static_cast<int64_t>(i) * 40'000'000, 0, 0, i, static_cast<int>(i)));
        }
        auto batches = assembler.push(std::move(frames));
        auto last = assembler.flush_eos();

        mv::NaiveTracker tracker;
        std::vector<std::string> keys_per_frame;
        for (const auto& batch : batches) {
            for (const auto& f : batch.frames) {
                auto tracked = tracker.update(f.presentation_index, f.pts_ns,
                                              {detections[static_cast<size_t>(f.presentation_index)]});
                keys_per_frame.push_back(tracked[0].local_track_key);
            }
        }
        if (last) {
            for (const auto& f : last->frames) {
                auto tracked = tracker.update(f.presentation_index, f.pts_ns,
                                              {detections[static_cast<size_t>(f.presentation_index)]});
                keys_per_frame.push_back(tracked[0].local_track_key);
            }
        }
        return keys_per_frame;
    };

    auto keys1 = run_with_batch_size(1);
    auto keys4 = run_with_batch_size(4);
    auto keys8 = run_with_batch_size(8);

    CHECK(keys1.size() == 12, "batch 1 produced 12 frame results");
    CHECK(keys4.size() == 12, "batch 4 produced 12 frame results");
    CHECK(keys8.size() == 12, "batch 8 produced 12 frame results");
    CHECK(keys1 == keys4, "batch 1 and batch 4 track keys match");
    CHECK(keys4 == keys8, "batch 4 and batch 8 track keys match");
    return true;
}

// ---------------------------------------------------------------------------

int main() {
    std::cout << std::unitbuf;
    std::cerr << std::unitbuf;
    std::cout << "Running M5.1 sequence contract tests...\n";
    g_failures = 0;

    auto run = [](const char* name, bool (*fn)()) {
        int before = g_failures;
        bool ok = fn();
        if (ok && g_failures == before) {
            std::cout << "[PASS] " << name << "\n";
        } else {
            std::cout << "[FAIL] " << name << "\n";
        }
    };

    run("A. sequential_assembler", test_sequential_assembler);
    run("B. irregular_mux_buffers", test_irregular_mux_buffers);
    run("C. pts_ordering", test_pts_ordering);
    run("D. pts_regression", test_pts_regression);
    run("E. sampling", test_sampling);
    run("F. owner_lifetime", test_owner_lifetime);
    run("G. detector_mapping", test_detector_mapping);
    run("H. recognition_mapping", test_recognition_mapping);
    run("I. track_batch_boundary", test_track_batch_boundary);
    run("J. chunk_invariance", test_chunk_invariance);

    std::cout << "\nTotal failures: " << g_failures << "\n";
    return g_failures == 0 ? 0 : 1;
}

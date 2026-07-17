#pragma once

#include "mv/video/detection_mapper.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace mergenvision::video {

// Job-local deterministic track key allocator. Not thread-safe and not shared
// across jobs — exactly one instance per video job.
class LocalTrackAllocator {
public:
    LocalTrackAllocator() = default;

    std::string next() {
        ++counter_;
        std::ostringstream oss;
        oss << "RT" << std::setw(6) << std::setfill('0') << counter_;
        return oss.str();
    }

    int allocated_count() const { return counter_; }

private:
    int counter_ = 0;
};

enum class TrackState { kTentative, kConfirmed, kLost, kRemoved };

struct TrackedDetection {
    FaceDetection detection;
    std::string local_track_key;
};

struct Track {
    std::string key;
    BBox last_bbox;
    int64_t last_pts_ns = 0;
    uint64_t last_presentation_index = 0;
    TrackState state = TrackState::kTentative;
    int consecutive_hits = 1;
    int consecutive_misses = 0;
};

// Minimal deterministic tracker adapter used to prove the sequential chronology
// and batch-boundary contracts. It is intentionally simple and will later be
// replaced or extended by the production Python/C++ tracker; for M5.1 its job
// is to guarantee that track identity survives detector batch boundaries and
// that rerun with identical input/config is deterministic.
class NaiveTracker {
public:
    explicit NaiveTracker(float association_distance_threshold = 50.0f,
                          int confirmation_threshold = 1,
                          int max_consecutive_misses = 5)
        : threshold_(association_distance_threshold),
          confirmation_threshold_(confirmation_threshold),
          max_consecutive_misses_(max_consecutive_misses) {}

    std::vector<TrackedDetection> update(
        uint64_t presentation_index,
        int64_t pts_ns,
        const std::vector<FaceDetection>& detections) {
        std::vector<TrackedDetection> result;

        // Same-frame cannot-link and one-to-one matching are enforced by a
        // greedy assignment: each detection gets at most one track, each track
        // gets at most one detection in this frame. New tentative tracks are
        // staged separately so the |track_matched| bitmap stays valid while
        // matching existing tracks.
        std::vector<bool> track_matched(tracks_.size(), false);
        std::vector<Track> new_tracks;
        std::vector<FaceDetection> sorted_detections = detections;
        std::sort(sorted_detections.begin(), sorted_detections.end(),
                  [](const FaceDetection& a, const FaceDetection& b) {
                      if (a.frame.presentation_index != b.frame.presentation_index) {
                          return a.frame.presentation_index < b.frame.presentation_index;
                      }
                      return a.detection_ordinal < b.detection_ordinal;
                  });

        for (const auto& det : sorted_detections) {
            float best_dist = std::numeric_limits<float>::max();
            size_t best_track = static_cast<size_t>(-1);
            for (size_t t = 0; t < tracks_.size(); ++t) {
                if (track_matched[t] || tracks_[t].state == TrackState::kRemoved) {
                    continue;
                }
                float dist = bbox_center_distance(tracks_[t].last_bbox, det.bbox);
                if (dist < best_dist && dist <= threshold_) {
                    best_dist = dist;
                    best_track = t;
                }
            }
            if (best_track != static_cast<size_t>(-1)) {
                track_matched[best_track] = true;
                update_track(tracks_[best_track], det, presentation_index, pts_ns);
                result.push_back({det, tracks_[best_track].key});
            } else {
                Track nt;
                nt.key = allocator_.next();
                nt.last_bbox = det.bbox;
                nt.last_pts_ns = pts_ns;
                nt.last_presentation_index = presentation_index;
                nt.state = TrackState::kTentative;
                nt.consecutive_hits = 1;
                new_tracks.push_back(nt);
                result.push_back({det, nt.key});
            }
        }

        // Mark unmatched existing tracks as missed.
        for (size_t t = 0; t < tracks_.size(); ++t) {
            if (!track_matched[t] && tracks_[t].state != TrackState::kRemoved) {
                ++tracks_[t].consecutive_misses;
                if (tracks_[t].consecutive_misses > max_consecutive_misses_) {
                    tracks_[t].state = TrackState::kLost;
                }
            }
        }

        tracks_.insert(tracks_.end(), new_tracks.begin(), new_tracks.end());

        last_presentation_index_ = presentation_index;
        return result;
    }

    const std::vector<Track>& tracks() const { return tracks_; }

private:
    static float bbox_center_distance(const BBox& a, const BBox& b) {
        float cx_a = (a.x1 + a.x2) * 0.5f;
        float cy_a = (a.y1 + a.y2) * 0.5f;
        float cx_b = (b.x1 + b.x2) * 0.5f;
        float cy_b = (b.y1 + b.y2) * 0.5f;
        float dx = cx_a - cx_b;
        float dy = cy_a - cy_b;
        return std::sqrt(dx * dx + dy * dy);
    }

    void update_track(Track& track, const FaceDetection& det,
                      uint64_t presentation_index, int64_t pts_ns) {
        track.last_bbox = det.bbox;
        track.last_pts_ns = pts_ns;
        track.last_presentation_index = presentation_index;
        track.consecutive_misses = 0;
        ++track.consecutive_hits;
        if (track.state == TrackState::kTentative &&
            track.consecutive_hits >= confirmation_threshold_) {
            track.state = TrackState::kConfirmed;
        }
    }

    float threshold_ = 50.0f;
    int confirmation_threshold_ = 1;
    int max_consecutive_misses_ = 5;
    LocalTrackAllocator allocator_;
    std::vector<Track> tracks_;
    uint64_t last_presentation_index_ = 0;
};

}  // namespace mergenvision::video

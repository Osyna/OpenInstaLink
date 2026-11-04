//! Session-only results retain compressed source pixels and shared, exact model output.

use std::collections::VecDeque;
use std::io::Cursor;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result, ensure};
use image::codecs::png::{CompressionType, FilterType, PngEncoder};
use image::{ExtendedColorType, ImageEncoder, ImageFormat, ImageReader, Limits};

use super::{Analysis, MAX_PIXELS, RawOutput, interpret};
use crate::media::Frame;

pub struct Snapshot {
    pub model: String,
    pub sequence: u64,
    pub width: u32,
    pub height: u32,
    pub elapsed: Duration,
    source_png: Vec<u8>,
    output: RawOutput,
}

impl Snapshot {
    /// Compress on the analysis worker, never on the UI thread.
    pub fn capture(analysis: &Analysis) -> Result<Self> {
        let source = &analysis.source;
        ensure!(
            source.width > 0
                && source.height > 0
                && u64::from(source.width) * u64::from(source.height) <= MAX_PIXELS,
            "invalid history source dimensions"
        );
        ensure!(
            source.rgb.len() as u64 == u64::from(source.width) * u64::from(source.height) * 3,
            "history source RGB size does not match dimensions"
        );
        let mut source_png = Vec::new();
        PngEncoder::new_with_quality(&mut source_png, CompressionType::Fast, FilterType::Adaptive)
            .write_image(
                &source.rgb,
                source.width,
                source.height,
                ExtendedColorType::Rgb8,
            )
            .context("cannot losslessly compress history source PNG")?;
        Ok(Self {
            model: analysis.model.clone(),
            sequence: source.sequence,
            width: source.width,
            height: source.height,
            elapsed: analysis.output.elapsed,
            source_png,
            output: analysis.output.clone(),
        })
    }

    /// Decode only the selected entry off the UI thread; interpretation never invokes a model.
    pub fn restore(&self) -> Result<Analysis> {
        let mut reader =
            ImageReader::with_format(Cursor::new(self.source_png.as_slice()), ImageFormat::Png);
        let mut limits = Limits::default();
        limits.max_image_width = Some(self.width);
        limits.max_image_height = Some(self.height);
        limits.max_alloc = Some(u64::from(self.width) * u64::from(self.height) * 8 + 1024 * 1024);
        reader.limits(limits);
        let source = reader
            .decode()
            .context("cannot decode history source PNG")?
            .into_rgb8();
        ensure!(
            source.dimensions() == (self.width, self.height),
            "history source dimensions do not match the result"
        );
        interpret(
            &self.model,
            Arc::new(Frame {
                width: self.width,
                height: self.height,
                rgb: source.into_raw(),
                sequence: self.sequence,
            }),
            self.output.clone(),
        )
    }

    pub fn raw(&self) -> &RawOutput {
        &self.output
    }
}

#[derive(Clone)]
pub struct Record {
    pub id: u64,
    pub snapshot: Arc<Snapshot>,
}

#[derive(Default)]
pub struct History {
    entries: VecDeque<Record>,
    last_id: u64,
}

impl History {
    pub fn push(&mut self, snapshot: Arc<Snapshot>) -> Record {
        self.last_id = self.last_id.checked_add(1).expect("history ID exhausted");
        let record = Record {
            id: self.last_id,
            snapshot,
        };
        if self.entries.len() == 100 {
            self.entries.pop_back();
        }
        self.entries.push_front(record.clone());
        record
    }

    pub fn entries(&self) -> &VecDeque<Record> {
        &self.entries
    }

    pub fn get(&self, id: u64) -> Option<&Record> {
        self.entries.iter().find(|record| record.id == id)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn clear(&mut self) {
        self.entries.clear();
    }

    pub fn stored_bytes(&self) -> usize {
        self.entries
            .iter()
            .map(|record| record.snapshot.source_png.len() + record.snapshot.output.bytes.len())
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};

    fn analysis(model: &str, sequence: u64, payload: Value) -> Analysis {
        let source = Arc::new(Frame {
            width: 32,
            height: 24,
            rgb: (0..32 * 24 * 3)
                .map(|i| (i as u64 + sequence * 17) as u8)
                .collect(),
            sequence,
        });
        let output = RawOutput {
            job_id: format!("model-{model}-{sequence}"),
            content_type: "application/json".into(),
            bytes: Arc::new(serde_json::to_vec(&payload).unwrap()),
            elapsed: Duration::from_millis(sequence),
            warning: Some("preserved runtime warning".into()),
        };
        interpret(model, source, output).unwrap()
    }

    fn caption(sequence: u64) -> Analysis {
        analysis(
            "florence-2-base",
            sequence,
            json!({"model":"florence-2-base","image":{"width":32,"height":24},"text":"a room","task":"caption"}),
        )
    }

    #[test]
    fn newest_hundred_have_stable_ids_across_eviction_and_clear() {
        let mut history = History::default();
        for sequence in 0..105 {
            let record = history.push(Arc::new(Snapshot::capture(&caption(sequence)).unwrap()));
            assert_eq!(record.id, sequence + 1);
        }
        assert_eq!(history.len(), 100);
        assert_eq!(
            history
                .entries()
                .iter()
                .map(|record| record.id)
                .collect::<Vec<_>>(),
            (6..=105).rev().collect::<Vec<_>>()
        );
        assert!(history.get(5).is_none());
        assert_eq!(history.get(6).unwrap().snapshot.sequence, 5);
        let expected_bytes: usize = history
            .entries()
            .iter()
            .map(|record| record.snapshot.source_png.len() + record.snapshot.raw().bytes.len())
            .sum();
        assert_eq!(history.stored_bytes(), expected_bytes);
        history.clear();
        assert_eq!(history.len(), 0);
        assert_eq!(history.stored_bytes(), 0);
        let record = history.push(Arc::new(Snapshot::capture(&caption(0)).unwrap()));
        assert_eq!(
            record.id, 106,
            "IDs must not reuse camera sequence or reset on clear"
        );
    }

    #[test]
    fn held_evicted_result_restores_after_clear_without_retaining_decoded_frames() {
        let mut history = History::default();
        let original = caption(71);
        let source = Arc::downgrade(&original.source);
        let annotated = Arc::downgrade(&original.annotated);
        let expected_source = original.source.rgb.clone();
        let expected_annotated = original.annotated.rgb.clone();
        let raw = original.output.bytes.clone();
        let held = history.push(Arc::new(Snapshot::capture(&original).unwrap()));
        assert!(Arc::ptr_eq(&held.snapshot.raw().bytes, &raw));
        drop(original);
        assert!(source.upgrade().is_none());
        assert!(annotated.upgrade().is_none());
        for sequence in 0..100 {
            history.push(Arc::new(Snapshot::capture(&caption(sequence)).unwrap()));
        }
        assert!(history.get(held.id).is_none());
        history.clear();
        let restored = held.snapshot.restore().unwrap();
        assert_eq!(restored.source.sequence, 71);
        assert_eq!(restored.source.rgb, expected_source);
        assert_eq!(restored.annotated.rgb, expected_annotated);
        assert!(Arc::ptr_eq(&restored.output.bytes, &raw));
    }

    #[test]
    fn distinct_model_snapshots_restore_exact_source_raw_and_render_geometry() {
        let point = json!({"x":8.0,"y":8.0,"z":0.1,"visibility":0.9,"presence":0.9});
        let fixtures = [
            (
                "mediapipe-hand",
                json!({"hands":[{"handedness":"Left","score":0.9,"landmarks":vec![point.clone();21]}]}),
            ),
            (
                "mediapipe-pose",
                json!({"poses":[{"landmarks":vec![point.clone();33],"world_landmarks":vec![json!({"x":0.2,"y":-0.5,"z":0.7});33]}]}),
            ),
            (
                "mediapipe-face",
                json!({"faces":[{"landmarks":vec![point;478],"blendshapes":[{"name":"jawOpen","score":0.8}]}]}),
            ),
            (
                "sam2.1-tiny",
                json!({"masks":[{"score":0.8,"area_fraction":0.3,"bbox":{"x1":4,"y1":4,"x2":20,"y2":20},"polygon":[[4,4],[20,4],[20,20],[4,20]]}]}),
            ),
            (
                "florence-2-base",
                json!({"task":"ocr","regions":[{"text":"sign","box":{"x1":4,"y1":4,"x2":20,"y2":20},"quad":[[4,4],[20,4],[20,20],[4,20]]}]}),
            ),
            (
                "yolo11n",
                json!({"detections":[{"label":"person","score":0.9,"box":{"x1":4,"y1":4,"x2":20,"y2":20}}]}),
            ),
        ];
        let mut originals = Vec::new();
        for (index, (model, mut payload)) in fixtures.into_iter().enumerate() {
            payload["model"] = json!(model);
            payload["image"] = json!({"width":32,"height":24});
            payload["text"] = json!(format!("result {index}"));
            originals.push(analysis(model, index as u64 + 1, payload));
        }
        for (index, model) in ["depth-anything-v2-small", "marigold-depth-v1-1"]
            .into_iter()
            .enumerate()
        {
            let mut original = caption(index as u64 + 7);
            let mut png = Vec::new();
            PngEncoder::new(&mut png)
                .write_image(&original.source.rgb, 32, 24, ExtendedColorType::Rgb8)
                .unwrap();
            original.output.content_type = "image/png".into();
            original.output.bytes = Arc::new(png);
            originals.push(interpret(model, original.source, original.output).unwrap());
        }
        let snapshots: Vec<_> = originals
            .iter()
            .map(|original| Snapshot::capture(original).unwrap())
            .collect();
        for (original, snapshot) in originals.iter().zip(&snapshots).rev() {
            let restored = snapshot.restore().unwrap();
            assert_eq!(snapshot.model, original.model);
            assert_eq!(snapshot.sequence, original.source.sequence);
            assert_eq!((snapshot.width, snapshot.height), (32, 24));
            assert_eq!(snapshot.elapsed, original.output.elapsed);
            assert_eq!(restored.model, original.model);
            assert_eq!(restored.source.sequence, original.source.sequence);
            assert_eq!((restored.source.width, restored.source.height), (32, 24));
            assert_eq!(restored.source.rgb, original.source.rgb);
            assert_eq!(restored.annotated.sequence, original.source.sequence);
            assert_eq!(restored.annotated.rgb, original.annotated.rgb);
            assert_eq!(restored.text, original.text);
            assert_eq!(restored.output.job_id, original.output.job_id);
            assert_eq!(restored.output.content_type, original.output.content_type);
            assert_eq!(restored.output.elapsed, original.output.elapsed);
            assert_eq!(restored.output.warning, original.output.warning);
            assert!(Arc::ptr_eq(&restored.output.bytes, &original.output.bytes));
            assert_eq!(
                format!("{:?}", restored.data),
                format!("{:?}", original.data)
            );
            assert_eq!(
                restored.depth.as_ref().map(|frame| &frame.rgb),
                original.depth.as_ref().map(|frame| &frame.rgb)
            );
        }
    }

    #[test]
    fn malformed_source_is_an_explicit_capture_error() {
        let mut original = caption(1);
        original.source = Arc::new(Frame {
            width: 32,
            height: 24,
            rgb: vec![0; 3],
            sequence: 1,
        });
        let error = Snapshot::capture(&original).err().unwrap();
        assert!(error.to_string().contains("RGB size"));
    }
}

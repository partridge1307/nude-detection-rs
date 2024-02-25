use anyhow::Result;
use image::{GenericImageView, ImageBuffer, Rgba};
use ndarray_stats::QuantileExt;
use opencv::core::{Rect, Vector};
use std::path::Path;
use tract_onnx::{prelude::*, tract_core::internal::tract_smallvec::SmallVec};

use crate::model::Detection;

const SIZE: usize = 320;
const SIZE_U32: u32 = 320;

pub type Model = SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;
pub type TractResult = SmallVec<[TValue; 4]>;

pub struct NudeDetector {
    pub onnx_session: Model,
}

impl NudeDetector {
    /// Initialize model from `path`
    pub fn new() -> Result<Self> {
        let model = tract_onnx::onnx()
            .model_for_path("model.onnx")?
            .with_input_fact(0, f32::fact([1, 3, SIZE, SIZE]).into())?
            .into_optimized()?
            .into_runnable()?;

        Ok(NudeDetector {
            onnx_session: model,
        })
    }

    /// Detect nude from image specified in `path`
    pub fn detect<P: AsRef<Path>>(&self, path: P) -> Result<Vec<Detection>> {
        let (tensor, resize_factor, pad_top, pad_left) = read_image(path)?;

        let results: TractResult = self.onnx_session.run(tvec!(tensor.into()))?;

        let detections = post_process(results, resize_factor, pad_top, pad_left)?;

        Ok(detections)
    }
}

/// Read image from specific `path`.
///
/// Return `Tensor`, `resize_factor`, `top padding`, `left padding`
fn read_image<P: AsRef<Path>>(path: P) -> Result<(Tensor, f32, u32, u32)> {
    // Load image from path
    let img = image::open(path)?;
    let resized = img.resize(SIZE_U32, SIZE_U32, image::imageops::FilterType::Triangle);

    // Calculate resize factor
    let (img_width, img_height) = (img.width() as f32, img.height() as f32);
    let (resized_width, resized_height) = (resized.width() as f32, resized.height() as f32);
    let resize_factor = ((img_width.powi(2) + img_height.powi(2))
        / (resized_width.powi(2) + resized_height.powi(2)))
    .sqrt();

    // Calculate `x` axis padding, `y` axis padding
    let pad_x = SIZE_U32 - resized.width();
    let pad_y = SIZE_U32 - resized.height();

    // Calculate to make center
    let (pad_top, pad_bottom) = (pad_y / 2, pad_y / 2);
    let (pad_left, pad_right) = (pad_x / 2, pad_x / 2);

    // Create black padding color
    let padding_color: Rgba<u8> = Rgba([0, 0, 0, 255]);
    // Draw padding for image
    let image = ImageBuffer::from_fn(
        resized.width() + pad_left + pad_right,
        resized.height() + pad_top + pad_bottom,
        |x, y| {
            if x >= pad_left
                && x < resized.width() + pad_left
                && y >= pad_top
                && y <= resized.height() + pad_top
            {
                resized.get_pixel(x - pad_left, y - pad_top)
            } else {
                padding_color
            }
        },
    );

    // Turn into 4 dimensional, normalize image and convert into `Tensor`
    let tensor: Tensor =
        tract_ndarray::Array4::from_shape_fn((1, 3, SIZE, SIZE), |(_, c, y, x)| {
            image[(x as _, y as _)][c] as f32 / 255.0
        })
        .into();

    Ok((tensor, resize_factor, pad_top, pad_left))
}

fn post_process(
    results: TractResult,
    resize_factor: f32,
    pad_top: u32,
    pad_left: u32,
) -> Result<Vec<Detection>> {
    // Convert `3D` array into `2D` array
    let mut data = results[0]
        .to_array_view::<f32>()?
        .remove_axis(tract_ndarray::Axis(0));
    // Transpose array
    data.swap_axes(0, 1);

    // Create `boxes`, `scores`, `class_ids` container
    let mut boxes = Vector::new();
    let mut scores = Vector::new();
    let mut class_ids = Vec::new();
    for row in data.rows() {
        // Ignore first four value of row
        let classes_scores = row.slice(tract_ndarray::s![4..]);

        // Get max `score` in classes
        let max_score = classes_scores.max()?;

        if max_score >= &0.2 {
            // Get index of max `score` in class
            let class_id = classes_scores.argmax()?;
            let (left, top, w, h) = (row[0], row[1], row[2], row[3]);

            // Calculate position
            let x = ((left - w * 0.5 - pad_left as f32).round() * resize_factor) as i32;
            let y = ((top - h * 0.5 - pad_top as f32).round() * resize_factor) as i32;
            let width = (w * resize_factor).round() as i32;
            let height = (h * resize_factor).round() as i32;

            // Push to container
            class_ids.push(class_id);
            scores.push(*max_score);
            boxes.push(Rect::new(x, y, width, height));
        }
    }

    // Calculate NMS
    let mut indices = Vector::new();
    opencv::dnn::nms_boxes_def(&boxes, &scores, 0.25, 0.45, &mut indices)?;

    // Return `Detection`
    Ok(indices
        .iter()
        .map(|i| {
            let i = i as usize;

            Detection {
                class: class_ids[i]
                    .try_into()
                    .expect("Failed to get label. Class ID are not covered"),
                score: scores.get(i).expect("Failed to get score"),
                rect: boxes.get(i).expect("Failed to get box"),
            }
        })
        .collect())
}

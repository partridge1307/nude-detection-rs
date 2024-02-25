use nude_detection::NudeDetector;

fn main() {
    let nude_detector = NudeDetector::new().expect("Failed to load model");

    let result = nude_detector
        .detect("image.jpg")
        .expect("Failed to detect image");

    println!("{:?}", result)
}

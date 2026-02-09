use image::GrayImage;
use num::Complex;
use std::error::Error;

fn calc_mandelbrot(
    max_iters: usize,
    x_min: f64,
    x_max: f64,
    y_min: f64,
    y_max: f64,
    width: usize,
    height: usize,
) -> Vec<Vec<usize>> {
    let mut rows: Vec<Vec<usize>> = Vec::with_capacity(height);

    for img_y in 0..height {
        let mut row: Vec<usize> = Vec::with_capacity(width);
        for img_x in 0..width {
            let x_percent = img_x as f64 / width as f64;
            let y_percent = img_y as f64 / height as f64;

            let cx = x_min + (x_max - x_min) * x_percent;
            let cy = y_min + (y_max - y_min) * y_percent;

            let point = mandelbrot_rander_point(cx, cy, max_iters);
            row.push(point);
        }
        rows.push(row);
    }

    rows
}

fn mandelbrot_rander_point(cx: f64, cy: f64, max_iters: usize) -> usize {
    let mut z = Complex::new(0.0, 0.0);
    let c = Complex::new(cx, cy);

    for i in 0..=max_iters {
        if z.norm() > 2.0 {
            return i;
        }
        z = z * z + c;
    }

    max_iters
}

fn draw_mandelbrot(
    escaped: Vec<Vec<usize>>,
    width: u32,
    height: u32,
    max_iters: usize,
) -> Result<(), Box<dyn Error>> {
    let raw_img: Vec<u8> = escaped
        .iter()
        .map(|inner| {
            inner
                .iter()
                .map(|&x| {
                    if x == max_iters {
                        255u8
                    } else {
                        ((x as f32 / max_iters as f32) * 255.0) as u8
                    }
                })
                .collect::<Vec<u8>>()
        })
        .flatten()
        .collect();

    let img = match GrayImage::from_raw(width, height, raw_img) {
        Some(im) => im,
        None => {
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("Invalid raw_img size"),
            )))
        }
    };

    img.save("image.png")?;

    Ok(())
}

fn main() {
    let escaped = calc_mandelbrot(1000, -2.0, 1.0, -1.0, 1.0, 3840, 2160);

    let _ = match draw_mandelbrot(escaped, 1920, 1080, 1000) {
        Ok(_) => println!("Successed save image as \"image.png\""),
        Err(e) => println!("Handled error: {}", e),
    };
}

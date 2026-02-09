use image::GrayImage;
use num::Complex;
use rayon::{
    iter::{IndexedParallelIterator, ParallelIterator},
    slice::ParallelSliceMut,
};
use std::error::Error;

fn calc_mandelbrot(
    max_iters: usize,
    x_min: f64,
    x_max: f64,
    y_min: f64,
    y_max: f64,
    width: usize,
    height: usize,
) -> Vec<usize> {
    let mut buf: Vec<usize> = vec![0; width * height];

    buf.par_chunks_mut(width).enumerate().for_each(|(y, row)| {
        let cy = y_min + (y_max - y_min) * (y as f64 / height as f64);
        for (x, pixel) in row.iter_mut().enumerate() {
            let cx = x_min + (x_max - x_min) * x as f64 / width as f64;
            *pixel = mandelbrot_at_point(cx, cy, max_iters);
        }
    });

    buf
}

fn mandelbrot_at_point(cx: f64, cy: f64, max_iters: usize) -> usize {
    let mut z = Complex::new(0.0, 0.0);
    let c = Complex::new(cx, cy);

    for i in 0..=max_iters {
        if z.norm_sqr() > 4.0 {
            return i;
        }
        z = z * z + c;
    }

    max_iters
}

fn draw_mandelbrot(
    escaped: Vec<usize>,
    width: u32,
    height: u32,
    max_iters: usize,
) -> Result<(), Box<dyn Error>> {
    let raw_img: Vec<u8> = escaped
        .iter()
        .map(|&x| {
            if x == max_iters {
                255u8
            } else {
                ((x as f32 / max_iters as f32) * 255.0) as u8
            }
        })
        .collect::<Vec<u8>>();

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
    let escaped = calc_mandelbrot(1000, -2.0, 1.0, -0.84375, 0.84375, 3840, 2160);

    let _ = match draw_mandelbrot(escaped, 3840, 2160, 1000) {
        Ok(_) => println!("Successed save image as \"image.png\""),
        Err(e) => println!("Handled error: {}", e),
    };
}

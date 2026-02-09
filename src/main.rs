use clap::{Parser, Subcommand};
use image::GrayImage;
use num::Complex;
use rayon::{
    iter::{IndexedParallelIterator, ParallelIterator},
    slice::ParallelSliceMut,
};
use std::error::Error;

/// Parallel CPU-based Mandelbrot set generator (rayon crate).
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Number of iterations to check whether a point belongs to a set
    #[arg(short, long, default_value_t = 1000)]
    iters: usize,

    /// Width of result picture
    #[arg(short, long, default_value_t = 3840)]
    width: usize,

    /// Height of result picture
    #[arg(short, long, default_value_t = 2160)]
    height: usize,

    /// Minimum value of the X-axis for consideration on the complex plane
    #[arg(long, default_value_t = -2.0)]
    x_min: f64,

    /// Maximum value of the X-axis for consideration on the complex plane
    #[arg(long, default_value_t = 1.0)]
    x_max: f64,

    /// Minimum value of the Y-axis for consideration on the complex plane
    #[arg(long, default_value_t = -0.84375)]
    y_min: f64,

    /// Maximum value of the Y-axis for consideration on the complex plane}
    #[arg(long, default_value_t = 0.84375)]
    y_max: f64,

    #[command(subcommand)]
    location: Option<Location>,
}
#[derive(Subcommand, Debug)]
enum Location {
    /// Seahorse Valley (double spirals)
    Seahorse,
    /// Deep spiral zoom
    DeepSpiral,
    /// Elephant Valley
    Elephant,
}

impl Location {
    fn coords(&self, aspect: f64) -> (f64, f64, f64, f64) {
        match self {
            Location::Seahorse => {
                let (x_min, x_max) = (-0.7856455, -0.7340665);
                let dx = x_max - x_min;
                let cy = 0.12554725;
                let dy = dx / aspect;
                (x_min, x_max, cy - dy / 2.0, cy + dy / 2.0)
            }
            Location::DeepSpiral => {
                let (x_min, x_max) = (-0.745538, -0.743538);
                let dx = x_max - x_min;
                let cy = 0.121200;
                let dy = dx / aspect;
                (x_min, x_max, cy - dy / 2.0, cy + dy / 2.0)
            }
            Location::Elephant => {
                let (x_min, x_max) = (0.275, 0.28);
                let dx = x_max - x_min;
                let cy = 0.007;
                let dy = dx / aspect;
                (x_min, x_max, cy - dy / 2.0, cy + dy / 2.0)
            }
        }
    }
}

fn calc_mandelbrot(
    iters: usize,
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
            *pixel = mandelbrot_at_point(cx, cy, iters);
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
    iters: usize,
) -> Result<(), Box<dyn Error>> {
    let raw_img: Vec<u8> = escaped
        .iter()
        .map(|&x| {
            if x == iters {
                255u8
            } else {
                ((x as f32 / iters as f32) * 255.0) as u8
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
    let args = Args::parse();

    let aspect = args.width as f64 / args.height as f64;

    let (x_min, x_max, y_min, y_max) = if let Some(loc) = args.location {
        loc.coords(aspect)
    } else {
        (args.x_min, args.x_max, args.y_min, args.y_max)
    };

    let escaped = calc_mandelbrot(
        args.iters,
        x_min,
        x_max,
        y_min,
        y_max,
        args.width,
        args.height,
    );

    let _ = match draw_mandelbrot(escaped, args.width as u32, args.height as u32, args.iters) {
        Ok(_) => println!("Successed save image as \"image.png\""),
        Err(e) => println!("Handled error: {}", e),
    };
}

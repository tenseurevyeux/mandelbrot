use clap::{Parser, Subcommand};
use image::GrayImage;
use num::Complex;
use rayon::{
    iter::{IndexedParallelIterator, ParallelIterator},
    slice::ParallelSliceMut,
};
use std::error::Error;
use wide::{f64x4, CmpLe};

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

#[derive(Debug, Clone, Copy)]
struct Complex4 {
    real: f64x4,
    imag: f64x4,
}

fn calc_mandelbrot(
    iters: usize,
    x_min: f64,
    x_max: f64,
    y_min: f64,
    y_max: f64,
    width: usize,
    height: usize,
) -> Vec<u64> {
    let mut buf: Vec<u64> = vec![0; width * height];

    let dx = (x_max - x_min) / width as f64;
    let dy = (y_max - y_min) / height as f64;

    buf.par_chunks_mut(width).enumerate().for_each(|(y, row)| {
        let cy_val = y_min + (y as f64) * dy;
        let cy4 = f64x4::splat(cy_val);

        let mut chunks = row.chunks_exact_mut(4);
        let mut x = 0;

        let x_offset = f64x4::new([0.0, 1.0, 2.0, 3.0]);
        let dx4 = f64x4::splat(dx);
        let x_min4 = f64x4::splat(x_min);

        for chunk in chunks.by_ref() {
            let x_base = f64x4::splat(x as f64);
            let cx4 = x_min4 + (x_base + x_offset) * dx4;
            let c = Complex4 {
                real: cx4,
                imag: cy4,
            };
            let results = mandelbrot_at_vec(&c, iters);

            chunk[0] = results[0];
            chunk[1] = results[1];
            chunk[2] = results[2];
            chunk[3] = results[3];

            x += 4;
        }

        for pixel in chunks.into_remainder() {
            let cx = x_min + (x as f64) * dx;
            *pixel = mandelbrot_at_point(cx, cy_val, iters);
            x += 1;
        }
    });

    buf
}

#[unsafe(no_mangle)]
#[inline(never)]
fn mandelbrot_at_vec(c: &Complex4, iters: usize) -> [u64; 4] {
    let mut z = *c;
    let mut count = f64x4::splat(0.0);
    let threshold = f64x4::splat(4.0);

    for _ in 0..iters {
        let rr = z.real * z.real;
        let ii = z.imag * z.imag;

        let mask = (rr + ii).simd_le(threshold);

        if !mask.any() {
            break;
        }

        count += mask.blend(f64x4::splat(1.0), f64x4::splat(0.0));

        let ri = z.real * z.imag;
        z.real = rr - ii + c.real;
        z.imag = ri + ri + c.imag;
    }

    let arr: [f64; 4] = count.into();
    [arr[0] as u64, arr[1] as u64, arr[2] as u64, arr[3] as u64]
}

#[unsafe(no_mangle)]
#[inline(never)]
fn mandelbrot_at_point(cx: f64, cy: f64, iters: usize) -> u64 {
    let mut z = Complex::new(0.0, 0.0);
    let c = Complex::new(cx, cy);

    for i in 0..=iters {
        if z.norm_sqr() > 4.0 {
            return i as u64;
        }
        z = z * z + c;
    }

    iters as u64
}

fn draw_mandelbrot(
    escaped: Vec<u64>,
    width: u32,
    height: u32,
    iters: usize,
) -> Result<(), Box<dyn Error>> {
    let raw_img: Vec<u8> = escaped
        .iter()
        .map(|&x| {
            if x == iters as u64 {
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

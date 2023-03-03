extern crate claxon;
extern crate flac_bound;
extern crate samplerate;

use claxon::FlacReader;
use dasp::Sample;
use flac_bound::{FlacEncoder, WriteWrapper};
use samplerate::{ConverterType, Samplerate};
use std::env;
use std::fs::File;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() != 3 {
        println!("Usage: {} <input.flac> <output.flac>", args[0]);
        return;
    }

    let input_file = &args[1];
    let output_file = &args[2];

    let mut reader = FlacReader::open(input_file).unwrap();
    let stream_info = reader.streaminfo();

    println!("{:?}", stream_info);

    let mut outf = File::create(output_file).unwrap();
    let mut outw = WriteWrapper(&mut outf);
    let mut enc = FlacEncoder::new()
        .unwrap()
        .channels(stream_info.channels)
        .bits_per_sample(stream_info.bits_per_sample)
        .compression_level(8)
        .sample_rate(44100) // default is 44100
        .init_write(&mut outw)
        .unwrap();

    // let target_sample_rate = stream_info.sample_rate;
    let target_sample_rate = 44100;
    let channels = stream_info.channels;

    let samples = reader
        .samples()
        .filter_map(Result::ok)
        .map(i32::to_sample::<f32>)
        .collect();

    let resampled_samples = resample(
        samples,
        stream_info.sample_rate,
        target_sample_rate,
        stream_info.channels as usize,
    );

    let new_samples = resampled_samples
        .iter()
        .map(|x| to_i32_sample(*x))
        .collect::<Vec<i32>>();

    println!("Signals converted to target sample rate");

    enc.process_interleaved(new_samples.as_slice(), (new_samples.len() / (channels as usize)) as u32).unwrap();

    let _ = enc.finish();
}

fn to_i32_sample(mut f32_sample: f32) -> i32 {
    f32_sample = f32_sample.clamp(-1.0, 1.0);
    if f32_sample >= 0.0 {
        ((f32_sample as f32 * i32::MAX as f32) + 0.5) as i32
    } else {
        ((-f32_sample as f32 * i32::MIN as f32) - 0.5) as i32
    }
}


fn resample(samples: Vec<f32>, input_rate: u32, output_rate: u32, channels: usize) -> Vec<f32> {
    let converter = Samplerate::new(
        ConverterType::SincBestQuality,
        input_rate,
        output_rate,
        channels,
    )
    .unwrap();
    converter.process(&samples).unwrap()
}

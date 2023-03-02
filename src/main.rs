extern crate claxon;
extern crate flac_bound;
extern crate samplerate;

use claxon::FlacReader;
use dasp::{interpolate::sinc::Sinc, ring_buffer, signal, Sample, Signal};
use flac_bound::{FlacEncoder, WriteWrapper};
use samplerate::{ConverterType, Samplerate};
use std::env;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};

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

    let target_sample_rate = 44100;
    let channels = stream_info.channels;

    // let old_samples = reader.samples().map(|s| s.unwrap()).collect::<Vec<i32>>();
    // let samples = reader.samples().map(|x| x.unwrap() as f32).collect::<Vec<f32>>();
    let samples = reader
        .samples()
        .filter_map(Result::ok)
        .map(i32::to_sample::<f64>);

    // Convert the signal's sample rate using `Sinc` interpolation.
    let ring_buffer = ring_buffer::Fixed::from([[0.0]; 100]);
    let sinc = Sinc::new(ring_buffer);
    let signal = signal::from_interleaved_samples_iter::<_, [f64; 1]>(samples);
    let mut new_signal = signal.from_hz_to_hz(
        sinc,
        stream_info.sample_rate as f64,
        target_sample_rate as f64,
    );

    println!("Signals converted to target sample rate");

    // test signal

    // let x = new_signal.next();

    // println!("{:?}", x);

    // let converted_samples = new_signal
    //     .into_interleaved_samples()
    //     .into_iter()
    //     .map(to_i32_sample)
    //     .collect::<Vec<_>>();

    // println!("Signals converted into interleaved samples");

    // enc.process_interleaved(
    //     converted_samples.as_slice(),
    //     (converted_samples.len() / (channels as usize)) as u32,
    // )
    // .unwrap();

    let mut buffer: Vec<i32> = vec![];

    for frame in new_signal.until_exhausted() {
        // do nothing
        //
        // println!("{:?}", frame);
        buffer.push(to_i32_sample(frame[0]));
    }

    println!("finished ...");

    // enc.process_interleaved(buffers)
    enc.process_interleaved(buffer.as_slice(), (buffer.len() / (channels as usize)) as u32).unwrap();

    // enc.process(new_samples);

    // println!("Encoder processed interleaved samples");

    // let new_samples = resample(
    //     samples,
    //     stream_info.sample_rate,
    //     target_sample_rate,
    //     stream_info.channels as usize,
    // );

    // let final_signal = new_samples.iter().map(|t| to_i32_sample(*t)).collect::<Vec<i32>>();

    // let signal = signal::from_interleaved_samples_iter(final_signal);

    // println!("{:?}", final_signal.len());

    // enc.process_interleaved(old_samples.as_slice(), (old_samples.len() / (channels as usize)) as u32).unwrap();
    // enc.process_interleaved(final_signal.as_slice(), (final_signal.len() / (channels as usize)) as u32).unwrap();
    // enc.process(samples.as_slice()).unwrap();
    let _ = enc.finish();
}

fn to_i32_sample(mut f64_sample: f64) -> i32 {
    f64_sample = f64_sample.clamp(-1.0, 1.0);
    if f64_sample >= 0.0 {
        ((f64_sample as f64 * i32::MAX as f64) + 0.5) as i32
    } else {
        ((-f64_sample as f64 * i32::MIN as f64) - 0.5) as i32
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
    // out.iter().map(|s| Sample::from(*s)).collect()
}

fn quantize(samples: Vec<f32>) -> Vec<u8> {
    samples.iter().map(|s| *s as u8).collect()
}

fn write_samples(writer: &mut BufWriter<File>, samples: Vec<u8>) -> usize {
    writer.write_all(&samples).unwrap();
    samples.len()
}

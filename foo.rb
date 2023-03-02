require 'open3'

input_file = '../rrmpeg/sample_5_0.flac'
output_file = 'output.flac'

# Use flacinfo to get the current sample rate and number of channels
stdout, stderr, status = Open3.capture3("flacinfo -s -n #{input_file}")
if status.success?
  sample_rate, channels = stdout.strip.split("\n").map(&:to_i)
else
  puts "Failed to get sample rate and channels: #{stderr}"
  exit 1
end

# Use sox to resample the file to 44.1 kHz and 8-bit
command = "sox #{input_file} -r 44100 -b 8 -c #{channels} #{output_file}"
stdout, stderr, status = Open3.capture3(command)
if status.success?
  puts "Resampled file saved to #{output_file}"
else
  puts "Failed to resample file: #{stderr}"
  exit 1
end

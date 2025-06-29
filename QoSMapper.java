import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.TimeZone;

import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.Mapper;

public class QoSMapper extends Mapper<LongWritable, Text, Text, Text> {

    private static final SimpleDateFormat sdf = new SimpleDateFormat("HH");

    @Override
    protected void setup(Context context) {
        sdf.setTimeZone(TimeZone.getTimeZone("UTC"));
    }

    public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
        String line = value.toString();

        // Skip header
        if (line.startsWith("id,timestamp") || line.trim().isEmpty()) return;

        String[] fields = line.split(",");

        try {
            long timestamp = Long.parseLong(fields[1]);
            String hour = sdf.format(new Date(timestamp * 1000));
            String area = fields[39];

            String rsrp = fields[3];
            String snr = fields[6];
            String speed = fields[25];
            String congestion = fields[38];
            String humidity = fields[32];
            String precipProb = fields[28];

            String outputValue = String.join(",", rsrp, snr, speed, congestion, humidity, precipProb, "1");

            context.write(new Text(area + "_" + hour), new Text(outputValue));
        } catch (Exception e) {
            // Skip malformed rows
        }
    }
}

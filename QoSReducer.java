import java.io.IOException;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.Reducer;

public class QoSReducer extends Reducer<Text, Text, Text, Text> {

    public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
        double sumRSRP = 0, sumSNR = 0, sumSpeed = 0, sumCongestion = 0, sumHumidity = 0, sumPrecip = 0;
        int count = 0;

        for (Text val : values) {
            String[] parts = val.toString().split(",");
            if (parts.length < 7) continue;

            try {
                // skip this record if any field is empty
                boolean valid = true;
                for (int i = 0; i < 7; i++) {
                    if (parts[i] == null || parts[i].trim().isEmpty()) {
                        valid = false;
                        break;
                    }
                }
                if (!valid) continue;

                sumRSRP += Double.parseDouble(parts[0]);
                sumSNR += Double.parseDouble(parts[1]);
                sumSpeed += Double.parseDouble(parts[2]);
                sumCongestion += Double.parseDouble(parts[3]);
                sumHumidity += Double.parseDouble(parts[4]);
                sumPrecip += Double.parseDouble(parts[5]);
                count += Integer.parseInt(parts[6]);

            } catch (NumberFormatException e) {
                // skip malformed record
                continue;
            }
        }

        if (count > 0) {
            String output = String.format("%.2f,%.2f,%.2f,%.2f,%.2f,%.2f",
                    sumRSRP / count, sumSNR / count, sumSpeed / count,
                    sumCongestion / count, sumHumidity / count, sumPrecip / count);
            context.write(key, new Text(output));
        }
    }
}

package yich.nn.loader;

import yich.base.util.BitUtil;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.zip.GZIPInputStream;

public class IDXFileParser {

    public static IDXObject parse(String path) {
        if (!Files.exists(Paths.get(path))) {
            path = System.getProperty("user.dir") + File.separator + path;
        }
        try (InputStream fis = new FileInputStream(path);
             InputStream is = new GZIPInputStream(fis);) {
            byte[] magic;
            is.read(magic = new byte[4]);
            byte data_type = magic[2];
            int dim = (int) magic[3];

            int[] d_sizes = new int[dim];

            int total = 0;
            byte[] buf;
            for (int i = 0; i < dim; i++) {
                is.read(buf = new byte[4]);
                d_sizes[i] = BitUtil.bytesToInt(buf);
                total = total == 0 ? d_sizes[i] : total * d_sizes[i];
            }
            //System.out.println(total);

            byte[] data = is.readAllBytes();

            return new IDXObject(data_type, d_sizes, data);


        } catch (IOException e) {
            e.printStackTrace();
        }

        return null;
    }


}

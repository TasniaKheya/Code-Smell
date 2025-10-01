package com.example.eye_disease_classification;

import androidx.appcompat.app.AppCompatActivity;
import android.os.Bundle;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;
import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;

public class MainActivity extends AppCompatActivity {

    Module model;
    EditText inputCode;
    TextView outputText;
    Button predictBtn;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        inputCode = findViewById(R.id.inputCode);
        outputText = findViewById(R.id.outputText);
        predictBtn = findViewById(R.id.predictBtn);

        try {
            model = Module.load(assetFilePath("smell_detector_scripted.pt"));
        } catch (IOException e) {
            e.printStackTrace();
        }

        predictBtn.setOnClickListener(v -> {
            String code = inputCode.getText().toString();
            float[] embedding = generateFakeEmbedding(code);  // placeholder
            Tensor inputTensor = Tensor.fromBlob(embedding, new long[]{1, 768});
            Tensor output = model.forward(IValue.from(inputTensor)).toTensor();
            float[] scores = output.getDataAsFloatArray();


            // Apply sigmoid
            for (int i = 0; i < scores.length; i++) {
                scores[i] = 1 / (1 + (float)Math.exp(-scores[i]));
            }

            StringBuilder sb = new StringBuilder();
            String[] smells = {"Long Method", "Data Class", "Long Parameter List"};
            for (int i = 0; i < scores.length; i++) {
                sb.append(smells[i]).append(": ").append(scores[i] > 0.5 ? "Yes" : "No").append("\n");
            }
            outputText.setText(sb.toString());
        });
    }

    private float[] generateFakeEmbedding(String code) {
        // Simple placeholder: convert chars to normalized floats
        float[] emb = new float[768];
        for (int i = 0; i < 768; i++) {
            emb[i] = i < code.length() ? (code.charAt(i) / 128.0f) : 0;
        }
        return emb;
    }

    private String assetFilePath(String assetName) throws IOException {
        File file = new File(getFilesDir(), assetName);
        if (!file.exists()) {
            try (InputStream is = getAssets().open(assetName)) {
                byte[] buffer = new byte[is.available()];
                is.read(buffer);
                java.io.FileOutputStream os = new java.io.FileOutputStream(file);
                os.write(buffer);
                os.close();
            }
        }
        return file.getAbsolutePath();
    }
}

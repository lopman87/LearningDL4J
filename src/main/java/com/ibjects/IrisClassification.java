package com.ibjects;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.apache.log4j.BasicConfigurator;
public class IrisClassification {
    private static final int FEATURES_COUNT = 4;
    private static final int CLASSES_COUNT = 3;

    /**
     * https://medium.com/datactw/deep-learning-for-java-dl4j-getting-started-tutorial-2259c76c0a7c#id_token=eyJhbGciOiJSUzI1NiIsImtpZCI6IjZhMWQyNmQ5OTJiZTdhNGI2ODliZGNlMTkxMWY0ZTlhZGM3NWQ5ZjEiLCJ0eXAiOiJKV1QifQ.eyJpc3MiOiJodHRwczovL2FjY291bnRzLmdvb2dsZS5jb20iLCJuYmYiOjE2MjMyODg3NTgsImF1ZCI6IjIxNjI5NjAzNTgzNC1rMWs2cWUwNjBzMnRwMmEyamFtNGxqZGNtczAwc3R0Zy5hcHBzLmdvb2dsZXVzZXJjb250ZW50LmNvbSIsInN1YiI6IjEwOTI2MzE5MDQzODA0Mzc0OTY3MyIsImVtYWlsIjoiNTQwNzYxODk1bG9wbWFuQGdtYWlsLmNvbSIsImVtYWlsX3ZlcmlmaWVkIjp0cnVlLCJhenAiOiIyMTYyOTYwMzU4MzQtazFrNnFlMDYwczJ0cDJhMmphbTRsamRjbXMwMHN0dGcuYXBwcy5nb29nbGV1c2VyY29udGVudC5jb20iLCJuYW1lIjoiUGVuZyBMb25nIiwicGljdHVyZSI6Imh0dHBzOi8vbGgzLmdvb2dsZXVzZXJjb250ZW50LmNvbS9hL0FBVFhBSnludGgyZmoyV1UtdDkyb1B3VGhTM2JCZGUwbEJhRlE5U2ZJZEE1PXM5Ni1jIiwiZ2l2ZW5fbmFtZSI6IlBlbmciLCJmYW1pbHlfbmFtZSI6IkxvbmciLCJpYXQiOjE2MjMyODkwNTgsImV4cCI6MTYyMzI5MjY1OCwianRpIjoiZmRlYTIzYmFkOTY4MDNiNmFkYWRiNTM4YzI2MTRlZjc1YzM1NTI0NSJ9.HnnmS4_07zaIi2nGFwS5PFjHi_TRSoSscxueY6U_7VWXcQjdDEgd3d4RfvdwHLzqB9zwiy4Pkzi3xdbm8clk9ml16o0XgoV0FiRpV_qnfiiWBO_Nvhno5wOklml1JJjCCsMO1pVYCiTAYVfhkM3KzhwRs_WMbud6vWX1ttE4c9dbX_jge2aGVQkEE6NfXINCmErt2tqFDdKn7KX-nmO1Ro00ybgI6VMHlZ4I3MHEIk-OxkvFpPOSWkWxg9zWgr3sLl8gHXoIDw9zH0dLGw43JILByhcD1ZBK3Cd-eWQ1nZil95PvK2dxgLS0TfqBYRDkx1LMuNaWYTZinnBRdaofFw
     * @param args
     */
    public static void main(String[] args) {

        BasicConfigurator.configure();
        loadData();

    }

    private static void loadData() {
        try(RecordReader recordReader = new CSVRecordReader(0,',')) {
            recordReader.initialize(new FileSplit(
                    new ClassPathResource("iris.csv").getFile()
            ));

            DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader, 150, FEATURES_COUNT, CLASSES_COUNT);
            DataSet allData = iterator.next();
            allData.shuffle(123);

            DataNormalization normalizer = new NormalizerStandardize();
            normalizer.fit(allData);
            normalizer.transform(allData);

            SplitTestAndTrain testAndTrain = allData.splitTestAndTrain(0.65);
            DataSet trainingData = testAndTrain.getTrain();
            DataSet testingData = testAndTrain.getTest();

            irisNNetwork(trainingData, testingData);

        } catch (Exception e) {
            Thread.dumpStack();
            new Exception("Stack trace").printStackTrace();
            System.out.println("Error: " + e.getLocalizedMessage());
        }
    }

    private static void irisNNetwork(DataSet trainingData, DataSet testData) {

        MultiLayerConfiguration configuration = new NeuralNetConfiguration.Builder()
                .activation(Activation.SIGMOID)
                .weightInit(WeightInit.XAVIER)
                .updater(new Nesterovs(0.1, 0.9))
                .l2(0.0001)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(FEATURES_COUNT).nOut(3).build())
                .layer(1, new DenseLayer.Builder().nIn(3).nOut(3).build())
                .layer(2, new OutputLayer.Builder(
                        LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).activation(Activation.SOFTMAX)
                        .nIn(3).nOut(CLASSES_COUNT).build())
                .backprop(true).pretrain(false)
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(configuration);
        model.init();
        model.fit(trainingData);

        INDArray output = model.output(testData.getFeatureMatrix());
        Evaluation eval = new Evaluation(3);
        eval.eval(testData.getLabels(), output);
        System.out.println(eval.stats());

    }
}

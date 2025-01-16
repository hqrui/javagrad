import java.util.Random;

public class Main {

    private static Value[] generatePoints(int dim, double minVal, double maxVal){
        Value[] v = new Value[dim];
        Random r = new Random();
        for(int i = 0; i < dim; i++){
            v[i] = new Value(minVal + r.nextDouble() * (maxVal - minVal));
        }
        return v;
    }

    private static Value isInEllipse(Value[] point, double h, double k, double rx, double ry){
        assert(point.length == 2);
        double x = point[0].getData();
        double y = point[1].getData();
        double in_reg = (x-h)*(x-h)/(rx*rx) + (y-k)*(y-k)/(ry*ry);
        return new Value((in_reg <= 1.0)? 1 : 0);
    }

    public static void main(String[] args) {
        final int SAMPLE_CNT = 1000;
        final int TEST_CNT = 100;
        final double MIN_VAL = -10;
        final double MAX_VAL = 10;
        final double h = 5;
        final double k = -2;
        final double rx = 7.37;
        final double ry = 9.91;
        Value[][] x = new Value[SAMPLE_CNT][];
        Value[] y = new Value[SAMPLE_CNT];
        for(int i = 0; i < SAMPLE_CNT; i++){
            x[i] = generatePoints(2, MIN_VAL, MAX_VAL);
            y[i] = isInEllipse(x[i], h, k, rx, ry);
//            System.out.println("Training point: "+ x[i][0].getData() + " " + x[i][1].getData() + ", inEclipse: " + y[i].getData());
        }
        Value[][] xt = new Value[TEST_CNT][];
        Value[] yt = new Value[TEST_CNT];
        for(int i = 0; i < TEST_CNT; i++){
            xt[i] = generatePoints(2, MIN_VAL, MAX_VAL);
            yt[i] = isInEllipse(xt[i], h, k, rx, ry);
//            System.out.println("Test point: "+ xt[i][0].getData() + " " + xt[i][1].getData() + ", inEclipse: " + yt[i].getData());
        }

        NeuralNetwork dnn = new NeuralNetwork(2, new int[]{10, 10, 4, 4, 1});// new int[]{1});

        Value[][] out; //Shld be in the format: [[pred1], [pred2]]
        Value loss;

        double stepSize = 0.5;
        for(int i=0;i<500;i++){
            out = dnn.run(x);
            double correct = 0, wrong = 0;
            for(int j = 0; j < SAMPLE_CNT; j ++){
                double pred = (out[j][0].getData() > 0.5) ? 1 : 0;
                if(pred  == y[j].getData())correct++;
                else wrong++;
            }
            if(i%100==99) stepSize/=2;
            loss = NeuralNetwork.mseLoss(out,y);
            loss.backward();
            System.out.println("Epoch " + (i+1) + ", Loss: "+ loss.getData());
            System.out.println("Correct: "+ correct + ", Wrong: " + wrong);
            dnn.step(stepSize);
            loss.zeroGrad();
        }

        System.out.println("Evaluating on test set:");
        out = dnn.run(xt);
        loss = NeuralNetwork.mseLoss(out,yt);
        System.out.println("Loss: "+ loss.getData());
        double correct = 0, wrong = 0;
        for(int j = 0; j < TEST_CNT; j ++){
            double pred = (out[j][0].getData() > 0.5) ? 1 : 0;
            if(pred  == yt[j].getData())correct++;
            else wrong++;
        }
        System.out.println("Correct: "+ correct + ", Wrong: " + wrong);
    }
}
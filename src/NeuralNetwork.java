public class NeuralNetwork {
    private final Layer[] layers;

    public NeuralNetwork(int inputLength, int[] layerSizes){
        layers = new Layer[layerSizes.length];
        layers[0] = new Layer(inputLength, layerSizes[0]);
        for(int i = 1; i < layerSizes.length; i++){
            layers[i] = new Layer(layerSizes[i-1], layerSizes[i]);
        }
    }

//    public Value[] run(Value[] inputX){
//        Value[] curX = null;
//        for(Layer layer : layers){
//            if(curX == null)curX = layer.run(inputX);
//            else curX = layer.run(curX);
//        }
//        return curX;
//    }

    public Value[][] run(Value[][] inputX){
        Value[][] curX = new Value[inputX.length][];
        for(int i = 0; i < inputX.length; i++){
            curX[i] = null;
            for(Layer layer : layers){
                if(curX[i] == null)curX[i] = layer.run(inputX[i]);
                else curX[i] = layer.run(curX[i]);
            }
        }
        return curX;
    }

    public void step(double stepSize){
        for(Layer layer : layers){
            layer.step(stepSize);
        }
    }

    public static Value mseLoss(Value[][] x, Value[] y){
        assert(x.length == y.length);
        Value totalLoss = new Value(0.0);
        for(int i = 0; i < x.length; i++){
            assert(x[i].length == 1); //Last layer should be only 1 neuron for binary MSE loss
            Value diff = y[i].add(x[i][0].neg());
            totalLoss = totalLoss.add(diff.pow(2));
        }
        Value denominatorReciprocal = new Value(1.0/x.length);
        return totalLoss.mul(denominatorReciprocal);
    }
}
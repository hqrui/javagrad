public class Layer {
    private final Neuron[] neurons;
    private final int inputLength;
    private final int outputLength;

    public Layer(int inputLength, int outputLength){
        this.inputLength = inputLength;
        this.outputLength = outputLength;
        neurons = new Neuron[outputLength];
        for(int i = 0;i < outputLength; i++){
            neurons[i] = new Neuron(inputLength);
        }
    }

    public Value[] run(Value[] x){
        assert(x.length == inputLength);
        Value[] out = new Value[outputLength];
        for(int i = 0; i < outputLength; i++){
            out[i] = neurons[i].run(x);
        }
        return out;
    }

    public void step(double stepSize){
        for(Neuron n : neurons){
            n.step(stepSize);
        }
    }
}
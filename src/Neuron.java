import java.util.Random;

public class Neuron {
    private Value b;
    private final Value[] w;

    public Neuron(int inputLength){
        assert(inputLength > 0);
        Random rng = new Random();
        w = new Value[inputLength];
        for(int i = 0; i < inputLength; i++){ //Initialise weights to random doubles between -1 and 1
            w[i] = new Value(-1.0 + rng.nextDouble() * 2.0);
        }
        b = new Value(-1.0 + rng.nextDouble() * 2.0);
    }

    public Value run(Value[] x){
        assert(x != null && x.length == w.length); //Inputs and weights must be the same length (not the more general shape as this is 1D)
        Value z = b.add(w[0].mul(x[0]));
        for(int i = 1; i < x.length; i++){
            z = z.add(w[i].mul(x[i]));
        }
        return z.tanh();
    }

    public void step(double stepSize){
        b.step(stepSize);
        for (Value v : w) {
            v.step(stepSize);
        }
    }
}
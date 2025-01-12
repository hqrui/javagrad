public class Value {
    private double data = 0.0;
    private double grad = 0.0;
    private Value child1 = null;
    private Value child2 = null;
    private Runnable backward = null;

    public Value(double _data){
        data = _data;
    }

    public Value(double _data, Value _child1, Value _child2){
        data = _data;
        child1 = _child1;
        child2 = _child2;
    }

    public void set_data(double _data){data = _data;}
    public double get_data(){return data;}

    public void accum_grad(double _delta_grad){grad += _delta_grad;}
    public double get_grad(){return grad;}

    public Value get_child1(){return child1;}
    public Value get_child2(){return child2;}

    public void set_backward_runnable(Runnable _backward){backward = _backward;}
    public void backward(){backward.run();}

    public Value add(Value other){
        Value out = new Value(data + other.get_data(), this, other);
        Runnable _backward = new Runnable() {
            public void run() {
                other.accum_grad(out.grad);
                accum_grad(out.grad);
            }
        };
        out.set_backward_runnable(_backward);
        return out;
    }



}

import java.util.ArrayList;
import java.util.Collections;

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

//    public void set_data(double _data){data = _data;}
    public double get_data(){return data;}

    public void accumulate_grad(double _delta_grad){grad += _delta_grad;}
    public double get_grad(){return grad;}

    public Value get_child1(){return child1;}
    public Value get_child2(){return child2;}

//    public void set_backward_runnable(Runnable _backward){backward = _backward;}
//    public void backward_single_op(){backward.run();}

    public void backward(){
        ArrayList<Value> nodes = new ArrayList<>(); //Topological sort
        backward(nodes);
        Collections.reverse(nodes);

        grad = 1.0;
        for(Value node: nodes){ //Backpropagate gradients
            node.backward.run();
            System.out.println(node.data);
        }
    }

    private void backward(ArrayList<Value> nodes){
        if(child1 != null && child1.backward != null) child1.backward(nodes); //Continue to backprop if there is a child which is not a leaf node
        if(child2 != null && child2.backward != null) child2.backward(nodes);
        nodes.add(this);
    }

    public void zero_grad(){
        grad = 0.0;
        if(child1 != null) child1.zero_grad();
        if(child2 != null) child2.zero_grad();
    }

    public Value add(Value other){
        Value out = new Value(data + other.get_data(), this, other);
        out.backward = () -> {
            grad += out.grad;
            other.grad += out.grad;
        };
        return out;
    }

    public Value mul(Value other){
        Value out = new Value(data * other.get_data(), this, other);
        out.backward = () -> {
            grad += other.data * out.grad;
            other.grad += data * out.grad;
        };
        return out;
    }



}

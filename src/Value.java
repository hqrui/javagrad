import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.Set;

public class Value {
    private double data;
    private double grad = 0.0;
    private final Value child1;
    private final Value child2;
    private Runnable backward = null;
    public String label;

    public Value(double data) {
        this.data = data;
        child1 = null;
        child2 = null;
        label = "Val";
    }

    public Value(double data, Value child1, Value child2, String label) {
        this.data = data;
        this.child1 = child1;
        this.child2 = child2;
        this.label = label;
    }

    public double getData() {
        return data;
    }

    public double getGrad() {
        return grad;
    }

    public Value getChild1() {
        return child1;
    }

    public Value getChild2() {
        return child2;
    }

    public void backward() { //There is a bug of multi visit to fix
        ArrayList<Value> nodes = new ArrayList<>(); //Topological sort
        Set<Value> visited = new HashSet<>();

        backward(nodes, visited);
        Collections.reverse(nodes);

        grad = 1.0;
//        System.out.println("NODES AND VISITED SIZE " + nodes.size() + " " + visited.size());
        for (Value node : nodes) { //Backpropagate gradients
            node.backward.run();
//            System.out.println("LABEL, DATA, GRAD, CHILDREN: " + node.label + " " + node.data + " " + node.grad + " " +  (((node.child1!=null)?1:0) + ((node.child2!=null)?1:0))); //node.child1.label +" " + node.child2.label);//
        }
    }

    private void backward(ArrayList<Value> nodes, Set<Value> visited) {
        assert (!visited.contains(this));
        visited.add(this);
        if (child1 != null && child1.backward != null && !visited.contains(child1)) child1.backward(nodes, visited); //Continue to backprop if there is a child which is not a leaf node
        if (child2 != null && child2.backward != null && !visited.contains(child2)) child2.backward(nodes, visited);
        nodes.add(this);
    }

    public void zeroGrad() {
        grad = 0.0;
        if (child1 != null) child1.zeroGrad();
        if (child2 != null) child2.zeroGrad();
    }

    public void step(double learningRate){
        data -= learningRate * grad;
    }

    public Value add(Value other) {
        Value out = new Value(data + other.getData(), this, other, "+");
        out.backward = () -> {
            grad += out.grad;
            other.grad += out.grad;
        };
        return out;
    }

    public Value mul(Value other) {
        Value out = new Value(data * other.getData(), this, other, "*");
        out.backward = () -> {
            grad += other.data * out.grad;
            other.grad += data * out.grad;
        };
        return out;
    }

    public Value pow(double other){
        Value out = new Value(Math.pow(data, other), this, null, "pow" + other);
        out.backward = () -> grad += other * Math.pow(data, other - 1) * out.grad;
        return out;
    }

    public Value neg(){
        Value out = new Value(-data, this, null, "neg");
        out.backward = () -> grad -= out.grad;
        return out;
    }

    public Value relu() {
        Value out = new Value((data > 0) ? data : 0, this, null, "relu");
        out.backward = () -> grad += (data > 0) ? out.grad : 0;
        return out;
    }

    public Value tanh() {
        Value out = new Value(Math.tanh(data), this, null, "tanh");
        out.backward = () -> grad += (1 - (out.data * out.data)) * out.grad;
        return out;
    }
}
public class Main {
    public static void main(String[] args) {
        System.out.println("Hello, World!");
        Value a = new Value(7);
        Value b = new Value(5);
        Value c = a.mul(b);
        Value d = new Value(3);
        Value e = c.mul(d);
        System.out.println(e.get_child1().get_child1().get_grad() + " " + e.get_child1().get_child2().get_grad() + " " + e.get_child1().get_grad() + " " +  e.get_child2().get_grad() + " " + e.get_grad());
        e.backward();
        System.out.println(e.get_child1().get_child1().get_grad() + " " + e.get_child1().get_child2().get_grad() + " " + e.get_child1().get_grad() + " " +  e.get_child2().get_grad() + " " + e.get_grad());
    }
}
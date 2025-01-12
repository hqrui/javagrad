public class Main {
    public static void main(String[] args) {
        System.out.println("Hello, World!");
        Value a = new Value(3.4);
        Value b = new Value(1.5);
        Value c = a.add(b);
        c.accum_grad(69);
        System.out.println(c.get_grad());
        System.out.println(c.get_child2().get_grad());
        c.backward();
        c.backward();
        System.out.println(c.get_grad());
        System.out.println(c.get_child1().get_grad());

    }
}
public class Main {
    public static void main(String[] args) {

        Value a = new Value(1);
        Value b = new Value(1);
        Value c = new Value(1);
        Value d = new Value(2);
        Value e = new Value(3);
        Value f = new Value(4);
        NeuralNetwork dnn = new NeuralNetwork(3, new int[]{3, 4, 2, 1});// new int[]{1});
        Value[] y = {new Value(1), new Value(0)};

        Value[][] x = {{a, b, c}, {d, e, f}};
        Value[][] out; //Shld be in the format: [[pred1], [pred2]]
        Value loss;

        double stepSize = 0.5;
        for(int i=0;i<20;i++){
            out = dnn.run(x);
            loss = NeuralNetwork.mseLoss(out,y);
            loss.backward();
            System.out.println("Loss: "+ loss.getData());
            dnn.step(stepSize);
            loss.zeroGrad();
        }

       /* Value a = new Value(7);
        Value b = new Value(5);
        Value c = a.add(b);
        Value d = new Value(3);
        Value e = c.mul(c);*/
//        Neuron k = new Neuron(5);
//        Value a = new Value(1);
//        Value b = new Value(1);
//        Value c = new Value(1);
//
//        Value[] x = {a, b, c};
//        Value out = k.run(x);
//
//
//        double stepSize = 0.1;
//        for(int i=0;i<10;i++){
//            out.backward();
//            System.out.println(out.getGrad() + " " + a.getGrad() + " " + b.getGrad() + " " + c.getGrad());
//            System.out.println("Loss: "+ out.data);
//            k.step(stepSize);
//            out.zeroGrad();
//            out = k.run(x);
//        }


//        System.out.println(out.getChild1().getChild1().getGrad() + " " + out.getChild1().getChild2().getGrad() + " " + out.getChild1().getGrad() + " "  + out.getGrad());
//        out.backward();
//        System.out.println(out.getChild1().getChild1().getGrad() + " " + out.getChild1().getChild2().getGrad() + " " + out.getChild1().getGrad() + " "  + out.getGrad());
    }
}
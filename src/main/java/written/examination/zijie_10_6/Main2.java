package written.examination.zijie_10_6;

import java.util.Stack;

public class Main2 {
    public static void main(String[] args) {
        boolean ans1 = isVaild("()");
        boolean ans2 = isVaild("()[]{}");
        boolean ans3 = isVaild("(]");
        boolean ans4 = isVaild("([}]");
        boolean ans5 = isVaild("{[]}");
        System.out.println(ans1);
        System.out.println(ans2);
        System.out.println(ans3);
        System.out.println(ans4);
        System.out.println(ans5);
    }

    public static boolean isVaild(String strs) {
        if (strs.length() % 2 != 0) return false;
        Stack<String> stack = new Stack<>();

        char[] charArray = strs.toCharArray();


        for (int i = 0; i < charArray.length; i++) {
            if (charArray[i] == '(') {
                stack.push(")");
            } else if (charArray[i] == '{') {
                stack.push(("}"));
            } else if (charArray[i] == '[') {
                stack.push("]");
            } else {
                if (stack.isEmpty()) return false;
                String top = stack.peek();
                if (!top.equals(charArray[i] + "")) {
                    return false;
                }
                stack.pop();

            }
        }
        return true;
    }
}

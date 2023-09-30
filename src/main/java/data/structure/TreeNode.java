package data.structure;

/**
 * @author 张壮
 * @description 力扣二叉树结构
 * @since 2023/2/22 22:06
 **/
public class TreeNode {


    public int val;
    public TreeNode left;
    public TreeNode right;

    public  TreeNode() {
    }

    public TreeNode(int val) {
        this.val = val;
    }

    public TreeNode(int val, TreeNode left, TreeNode right) {
        this.val = val;
        this.left = left;
        this.right = right;
    }

}


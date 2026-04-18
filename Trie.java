public class Trie {
        private class Node {
            Map<Character, Node> children = new HashMap<>();
            boolean isEndOfWord;

            public Node() {}
        }

        private Node root;

        public Trie() {
            root = new Node();
        }

        public void insert(String word) {
            Node node = root;
            for (char c : word.toCharArray()) {
                if (!node.children.containsKey(c)) {
                    node.children.put(c, new Node());
                }
                node = node.children.get(c);
            }
            node.isEndOfWord = true;
        }

        public boolean search(String word) {
            Node node = root;
            for (char c : word.toCharArray()) {
                if (!node.children.containsKey(c)) {
                    return false;
                }
                node = node.children.get(c);
            }
            return node.isEndOfWord; // Check if the last node is marked as isEndOfWord
        }

        public boolean startsWith(String prefix) {
            Node node = root;
            for (char c : prefix.toCharArray()) {
                if (!node.children.containsKey(c)) {
                    return false;
                }
                node = node.children.get(c);
            }
            return true;
        }
    }
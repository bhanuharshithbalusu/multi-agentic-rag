package com.debug;

import org.junit.Test;
import static org.junit.Assert.*;

public class TrieTest {
    @Test
    public void testTrie() {
        Trie trie = new Trie();
        trie.insert("apple");
        
        assertTrue("Should find complete word 'apple'", trie.search("apple"));
        assertFalse("Should NOT find prefix 'app' as a full word", trie.search("app"));
        assertTrue("startsWth 'app' should be true", trie.startsWith("app"));
        
        trie.insert("app");
        assertTrue("Should now find full word 'app'", trie.search("app"));
        
        assertFalse("Should not find 'beer'", trie.search("beer"));
    }
}

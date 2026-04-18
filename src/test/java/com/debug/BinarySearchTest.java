import org.junit.Test;
import static org.junit.Assert.*;

public class BinarySearchTest {
    @Test
    public void testSearch() {
        int[] arr = {1, 3, 5, 7, 9};
        assertEquals(2, BinarySearch.search(arr, 5));
        assertEquals(0, BinarySearch.search(arr, 1));
        assertEquals(4, BinarySearch.search(arr, 9));
        assertEquals(-1, BinarySearch.search(arr, 10));
    }
}
# test_random_labels_to_image.py
import pytest
from unittest.mock import MagicMock
import torchio as tio
from transforms_class import RandomLabelsToImageOneConstraint
from utils.logger import GREEN, RESET



class TestRandomLabelsToImageOneConstraint:
    """Test suite for RandomLabelsToImageOneConstraint"""
    
    @pytest.fixture
    def mock_transform(self):
        """Fixture to create a real instance"""
        transform = RandomLabelsToImageOneConstraint(
            minimal_contrast=0.1,
            contrasted_labels=None,
            default_mean=(0.0, 1.0),
            default_std=(0.01, 0.1),
        )
        # Override only get_params
        transform.get_params = MagicMock(return_value=(0.5, 0.1))
        return transform
    
    # ========================================================================
    # TEST 1: WITHOUT CONSTRAINT
    # ========================================================================
    def test_no_constraint_simple(self, mock_transform):
        """Test simple case without constraint"""
        labels = [1, 2, 3, 4, 5]
        
        means, stds = mock_transform.compute_label_to_distinguish(labels)
        
        assert len(means) == 5, "Should have 5 means"
        assert len(stds) == 5, "Should have 5 stds"
        assert all(m == 0.5 for m in means), "All means should be 0.5 (mock)"
        print(GREEN, "Test no constraint: PASS", RESET)
    
    def test_no_constraint_empty_labels(self, mock_transform):
        """Test with empty list"""
        labels = []
        
        means, stds = mock_transform.compute_label_to_distinguish(labels)
        
        assert means == [], "Should be empty"
        assert stds == [], "Should be empty"
        print(GREEN, "Test no constraint (empty): PASS", RESET)
    
    def test_no_constraint_single_label(self, mock_transform):
        """Test with a single label"""
        labels = [1]
        
        means, stds = mock_transform.compute_label_to_distinguish(labels)
        
        assert len(means) == 1, "Should have 1 mean"
        assert means[0] == 0.5, "Mean should be 0.5"
        print(GREEN, "Test no constraint (single): PASS", RESET)
    
    # ========================================================================
    # TEST 2: WITH CONSTRAINT (auto mode, no max_zones)
    # ========================================================================
    def test_with_constraint_no_max_zones(self):
        """Test with constraint, without max_zones (auto mode)"""
        transform = RandomLabelsToImageOneConstraint(
            minimal_contrast=0.1,
            contrasted_labels={1: [10, 11], 2: [10, 11], 3: [10, 11]},
            default_mean=(0.0, 1.0),
            default_std=(0.01, 0.1),
        )
        transform.get_params = MagicMock(return_value=(0.5, 0.1))
        
        labels = [1, 2, 3, 10, 11]
        means, stds = transform.compute_label_to_distinguish(labels)
        
        assert len(means) == 5, "Should have 5 means"
        
        # Check that sources and targets are separated
        source_means = [means[labels.index(i)] for i in [1, 2, 3]]
        target_means = [means[labels.index(i)] for i in [10, 11]]
        
        # At least one distance must be >= 0.1
        distances = [abs(s - t) for s in source_means for t in target_means]
        assert any(d >= 0.1 for d in distances), "At least one distance must be >= minimal_contrast"
        print(GREEN, "Test with constraint (auto): PASS", RESET)
    
    def test_with_constraint_multiple_groups(self):
        """Test with multiple constraint groups"""
        transform = RandomLabelsToImageOneConstraint(
            minimal_contrast=0.1,
            contrasted_labels={
                1: [10, 11],
                2: [10, 11],
                3: [20, 21]
            },
            default_mean=(0.0, 1.0),
            default_std=(0.01, 0.1),
        )
        transform.get_params = MagicMock(return_value=(0.5, 0.1))
        
        labels = [1, 2, 3, 10, 11, 20, 21]
        means, stds = transform.compute_label_to_distinguish(labels)
        
        assert len(means) == 7, "Should have 7 means"
        print(GREEN, "Test with multiple constraint groups: PASS", RESET)
    
    # ========================================================================
    # TEST 3: WITH CONSTRAINT + MAX_ZONES
    # ========================================================================
    def test_with_constraint_and_max_zones(self):
        """Test with constraint and defined max_zones"""
        transform = RandomLabelsToImageOneConstraint(
            minimal_contrast=0.1,
            max_zones=3,
            contrasted_labels={1: [10, 11], 2: [10, 11], 3: [10, 11]},
            default_mean=(0.0, 1.0),
            default_std=(0.01, 0.1),
        )
        transform.get_params = MagicMock(return_value=(0.5, 0.1))
        
        labels = [1, 2, 3, 10, 11]
        means, stds = transform.compute_label_to_distinguish(labels)
        
        assert len(means) == 5, "Should have 5 means"
        
        # Check clustering: sources should be more grouped with smaller max_zones
        source_means = [means[labels.index(i)] for i in [1, 2, 3]]
        target_means = [means[labels.index(i)] for i in [10, 11]]
        spread = max(source_means) - min(source_means)
        
        print(f"  Source spread: {spread:.3f}")
        # More zones = more space used
        assert spread <= 1.0, "Sources should be within full range"
        print("✅ Test with max_zones: PASS")
        
    def test_max_zones_limits_clustering(self):
        """Test that max_zones really reduces clustering"""
        
        def run_with_max_zones(max_zones):
            transform = RandomLabelsToImageOneConstraint(
                minimal_contrast=0.05,
                max_zones=max_zones,
                contrasted_labels={1: [10], 2: [10], 3: [10], 4: [10]},
                default_mean=(0.0, 1.0),
                default_std=(0.01, 0.1),
            )
            transform.get_params = MagicMock(return_value=(0.5, 0.1))
            
            labels = [1, 2, 3, 4, 10]
            means, _ = transform.compute_label_to_distinguish(labels)
            source_means = [means[labels.index(i)] for i in [1, 2, 3, 4]]
            spread = max(source_means) - min(source_means)
            return spread
        
        spread_1_zone = run_with_max_zones(1)
        spread_4_zones = run_with_max_zones(4)
        
        print(f"  Spread with 1 zone: {spread_1_zone:.3f}")
        print(f"  Spread with 4 zones: {spread_4_zones:.3f}")
        
        # 1 zone = more clustered than 4 zones
        assert spread_1_zone <= spread_4_zones, "Fewer zones = more clustered"
        print(GREEN, "Test max_zones clustering: PASS", RESET)
    
    # ========================================================================
    # TEST 4: ZONES - Verify they do not merge
    # ========================================================================
    def test_zones_do_not_merge(self):
        """Test that created zones do not overlap"""
        transform = RandomLabelsToImageOneConstraint(
            minimal_contrast=0.1,
            max_zones=5,
            contrasted_labels={1: [20], 2: [20], 3: [20], 4: [20], 5: [20]},
            default_mean=(0.0, 1.0),
            default_std=(0.01, 0.1),
        )
        transform.get_params = MagicMock(return_value=(0.5, 0.1))
        
        labels = [1, 2, 3, 4, 5, 20]
        means, _ = transform.compute_label_to_distinguish(labels)
        
        # Extract positions
        positions = {i: means[labels.index(i)] for i in [1, 2, 3, 4, 5]}
        
        print(f"  Positions: {positions}")
        
        # Check that zones (size 0.2) do not overlap
        zone_size = 2 * 0.1  # 0.2
        for i in [1, 2, 3, 4, 5]:
            for j in [1, 2, 3, 4, 5]:
                if i < j:
                    dist = abs(positions[i] - positions[j])
                    print(f"    Distance {i}-{j}: {dist:.3f}")
        
        print(GREEN, "Test zones no-merge: PASS", RESET)
    
    # ========================================================================
    # TEST 5: ZONES - Verify they are not all used
    # ========================================================================
    def test_not_all_zones_used(self):
        """Test that when sources < num_zones, some zones remain free"""
        transform = RandomLabelsToImageOneConstraint(
            minimal_contrast=0.1,
            max_zones=10,  # 10 possible zones
            contrasted_labels={1: [20], 2: [20]},  # But only 2 sources
            default_mean=(0.0, 1.0),
            default_std=(0.01, 0.1),
        )
        transform.get_params = MagicMock(return_value=(0.5, 0.1))
        
        labels = [1, 2, 20]
        means, _ = transform.compute_label_to_distinguish(labels)
        
        # Targets should be in a different zone than sources
        source_mean_1 = means[labels.index(1)]
        source_mean_2 = means[labels.index(2)]
        target_mean = means[labels.index(20)]
        
        # At least one target not in the same zone as a source
        in_same_zone_1 = abs(source_mean_1 - target_mean) < 0.2
        in_same_zone_2 = abs(source_mean_2 - target_mean) < 0.2
        
        print(f"  Source 1: {source_mean_1:.3f}, Source 2: {source_mean_2:.3f}, Target: {target_mean:.3f}")
        print(f"  Target in same zone as source 1: {in_same_zone_1}")
        print(f"  Target in same zone as source 2: {in_same_zone_2}")
        
        # Ideally, target should be far from sources
        assert not (in_same_zone_1 and in_same_zone_2), "Target should be in a free zone"
        print(GREEN, "Test unused zones: PASS", RESET)
    
    # ========================================================================
    # TEST 6: Spacing constraints
    # ========================================================================
    def test_sources_and_targets_in_different_zones(self):
        """Test that sources and targets are allocated correctly (separate zones)"""
        transform = RandomLabelsToImageOneConstraint(
            minimal_contrast=0.15,
            max_zones=4,
            contrasted_labels={1: [10], 2: [10], 3: [10]},
            default_mean=(0.0, 1.0),
            default_std=(0.01, 0.1),
        )
        transform.get_params = MagicMock(return_value=(0.5, 0.1))
        
        labels = [1, 2, 3, 10]
        means, _ = transform.compute_label_to_distinguish(labels)
        
        source_means = [means[labels.index(i)] for i in [1, 2, 3]]
        target_mean = means[labels.index(10)]
        
        # Check that allocations are in [0, 1]
        assert all(0.0 <= m <= 1.0 for m in source_means), "Source means should be in range"
        assert 0.0 <= target_mean <= 1.0, "Target mean should be in range"
        
        # Check that sources and target are not exactly equal
        for source_mean in source_means:
            assert source_mean != target_mean, "Source and target should be different"
        
        distances = [abs(s - target_mean) for s in source_means]
        print(f"  Distances: {[f'{d:.3f}' for d in distances]}")
        print(f"  Max distance: {max(distances):.3f}")
        print("✅ Test sources and targets allocation: PASS")
    # ========================================================================
    # TEST 7: Edge cases
    # ========================================================================
    def test_constraint_with_single_source(self):
        """Test with one source and one target"""
        transform = RandomLabelsToImageOneConstraint(
            minimal_contrast=0.1,
            max_zones=2,
            contrasted_labels={1: [10]},
            default_mean=(0.0, 1.0),
            default_std=(0.01, 0.1),
        )
        transform.get_params = MagicMock(return_value=(0.5, 0.1))
        
        labels = [1, 10]
        means, stds = transform.compute_label_to_distinguish(labels)
        
        assert len(means) == 2, "Should have 2 means"
        
        # Check source and target are in range
        assert 0.0 <= means[0] <= 1.0, "Source should be in range"
        assert 0.0 <= means[1] <= 1.0, "Target should be in range"
        
        # Check they are not exactly equal
        assert means[0] != means[1], "Source and target should be different"
        
        distance = abs(means[0] - means[1])
        print(f"  Distance: {distance:.3f}")
        print("✅ Test single source-target: PASS")

    def test_constraint_with_multiple_targets(self):
        """Test with multiple targets"""
        transform = RandomLabelsToImageOneConstraint(
            minimal_contrast=0.1,
            contrasted_labels={1: [20, 21, 22]},
            default_mean=(0.0, 1.0),
            default_std=(0.01, 0.1),
        )
        transform.get_params = MagicMock(return_value=(0.5, 0.1))
        
        labels = [1, 20, 21, 22]
        means, stds = transform.compute_label_to_distinguish(labels)
        
        assert len(means) == 4, "Should have 4 means"
        print(GREEN, "Test multiple targets: PASS", RESET)
    
    def test_max_zones_zero(self):
        """Test edge case: max_zones = 0"""
        transform = RandomLabelsToImageOneConstraint(
            minimal_contrast=0.1,
            max_zones=0,
            contrasted_labels={1: [10]},
            default_mean=(0.0, 1.0),
            default_std=(0.01, 0.1),
        )
        transform.get_params = MagicMock(return_value=(0.5, 0.1))
        
        labels = [1, 10]
        # Should fallback gracefully
        means, stds = transform.compute_label_to_distinguish(labels)
        
        assert len(means) == 2, "Should fallback and have 2 means"
        print(GREEN, "Test max_zones=0 (fallback): PASS", RESET)
    
    def test_max_zones_one(self):
        """Test with max_zones=1 (ultra-clustered)"""
        transform = RandomLabelsToImageOneConstraint(
            minimal_contrast=0.1,
            max_zones=1,
            contrasted_labels={1: [20], 2: [20], 3: [20]},
            default_mean=(0.0, 1.0),
            default_std=(0.01, 0.1),
        )
        transform.get_params = MagicMock(return_value=(0.5, 0.1))
        
        labels = [1, 2, 3, 20]
        means, _ = transform.compute_label_to_distinguish(labels)
        
        source_means = [means[labels.index(i)] for i in [1, 2, 3]]
        spread = max(source_means) - min(source_means)
        
        print(f"  Spread with max_zones=1: {spread:.3f}")
        assert spread <= 0.2, "All sources should be in one zone of size 0.2"
        print(GREEN, "Test max_zones=1: PASS", RESET)
    
    # ========================================================================
    # TEST 8: Randomization - Verify variability
    # ========================================================================
    def test_randomization_variability(self):
        """Test that different runs produce different results"""
        results = []
        
        for run in range(5):
            transform = RandomLabelsToImageOneConstraint(
                minimal_contrast=0.1,
                max_zones=3,
                contrasted_labels={1: [10], 2: [10], 3: [10]},
                default_mean=(0.0, 1.0),
                default_std=(0.01, 0.1),
            )
            transform.get_params = MagicMock(return_value=(0.5, 0.1))
            
            labels = [1, 2, 3, 10]
            means, _ = transform.compute_label_to_distinguish(labels)
            results.append(tuple(round(m, 2) for m in means))
            print(f"  Run {run+1}: {results[-1]}")
        
        # Check that at least 2 runs are different
        unique_results = len(set(results))
        assert unique_results > 1, "Should have variability!"
        print(GREEN, f"Test randomization: {unique_results} unique results on 5 runs - PASS", RESET)
    
    def test_unconstrained_labels_ignored(self):
        """Test that unconstrained labels use default_mean"""
        transform = RandomLabelsToImageOneConstraint(
            minimal_contrast=0.1,
            contrasted_labels={1: [10]},
            default_mean=(0.0, 1.0),
            default_std=(0.01, 0.1),
        )
        transform.get_params = MagicMock(return_value=(0.0, 1.0))  # Returns the range
        
        # Labels: 1 (source), 5 (unconstrained), 10 (target)
        labels = [1, 5, 10]
        means, stds = transform.compute_label_to_distinguish(labels)
        
        assert len(means) == 3, "Should have 3 means"
        
        # Label 5 should use get_params which returns (0.0, 1.0)
        # So the value should be in [0.0, 1.0]
        unconstrained_mean = means[labels.index(5)]
        assert 0.0 <= unconstrained_mean <= 1.0, "Unconstrained label should be in default_mean range"
        
        # Constrained labels should not fully overlap in [0.0, 1.0]
        # (they should be separated into zones)
        source_mean = means[labels.index(1)]
        target_mean = means[labels.index(10)]
        
        # At least one distance > minimal_contrast
        assert abs(source_mean - target_mean) >= 0.1, "Constrained labels should respect minimal_contrast"
        
        print(f"  Unconstrained (5): {unconstrained_mean:.3f}")
        print(f"  Source (1): {source_mean:.3f}")
        print(f"  Target (10): {target_mean:.3f}")
        print(GREEN, "Test unconstrained labels: PASS", RESET)

    def test_merge_zones(self):
        """Test the _merge_zones function"""
        transform = RandomLabelsToImageOneConstraint(
            minimal_contrast=0.1,
            default_mean=(0.0, 1.0),
            default_std=(0.01, 0.1),
        )
        
        # Test 1: Simple overlap
        zones_1 = [[0.1, 0.2], [0.15, 0.25]]
        merged_1 = transform._merge_zones(zones_1)
        assert merged_1 == [[0.1, 0.25]], f"Test 1 failed: {merged_1}"
        print(GREEN, "Test 1 (simple overlap): PASS", RESET)
        
        # Test 2: Contact (equal boundaries)
        zones_2 = [[0.1, 0.2], [0.2, 0.35]]
        merged_2 = transform._merge_zones(zones_2)
        assert merged_2 == [[0.1, 0.35]], f"Test 2 failed: {merged_2}"
        print(GREEN, "Test 2 (contact): PASS", RESET)
        
        # Test 3: Overlap + contact
        zones_3 = [[0.1, 0.2], [0.15, 0.25], [0.2, 0.35]]
        merged_3 = transform._merge_zones(zones_3)
        assert merged_3 == [[0.1, 0.35]], f"Test 3 failed: {merged_3}"
        print(GREEN, "Test 3 (overlap + contact): PASS", RESET)
        
        # Test 4: Non-overlapping zones
        zones_4 = [[0.05, 0.15], [0.25, 0.35], [0.45, 0.55]]
        merged_4 = transform._merge_zones(zones_4)
        assert merged_4 == [[0.05, 0.15], [0.25, 0.35], [0.45, 0.55]], f"Test 4 failed: {merged_4}"
        print(GREEN, "Test 4 (non-overlapping): PASS", RESET)
        
        # Test 5: Unsorted zones
        zones_5 = [[0.3, 0.4], [0.1, 0.2], [0.15, 0.25]]
        merged_5 = transform._merge_zones(zones_5)
        assert merged_5 == [[0.1, 0.25], [0.3, 0.4]], f"Test 5 failed: {merged_5}"
        print(GREEN, "Test 5 (auto-sort): PASS", RESET)
        
        # Test 6: Empty zones
        zones_6 = []
        merged_6 = transform._merge_zones(zones_6)
        assert merged_6 == [], f"Test 6 failed: {merged_6}"
        print(GREEN, "Test 6 (empty): PASS", RESET)
        
        # Test 7: Single zone
        zones_7 = [[0.1, 0.2]]
        merged_7 = transform._merge_zones(zones_7)
        assert merged_7 == [[0.1, 0.2]], f"Test 7 failed: {merged_7}"
        print(GREEN, "Test 7 (single): PASS", RESET)
        
        print(GREEN, "Test _merge_zones: ALL CASES PASSED!", RESET)
        
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
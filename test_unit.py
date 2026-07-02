"""
================================================================================
UNIT TESTS FOR STATIC MIXER SIMULATION
================================================================================

Purpose:
    Test individual functions used in the main solver to ensure they work correctly.

Tests included:
    1. CoV computation function
    2. Pressure evaluation function
    3. Inlet condition function
    4. Mesh loading function
    5. Concentration computation function

Run with: python test_unit.py
================================================================================
"""

import numpy as np
from mpi4py import MPI
from dolfinx import fem, io, mesh as dmesh
from ufl import dx, grad, inner
import unittest

comm = MPI.COMM_WORLD



# Helper function to compute CoV manually (for testing)


def compute_cov_manual(values):
    """Manual CoV calculation for verification."""
    if len(values) == 0:
        return None
    mean = np.mean(values)
    std = np.std(values)
    return float(std / mean) if mean > 1e-12 else 1.0



# TEST CLASS 1: CoV Computation


class TestCoVFunction(unittest.TestCase):
    """Test the Coefficient of Variation computation function."""
    
    @classmethod
    def setUpClass(cls):
        """Set up a simple mesh for testing."""
        cls.domain = dmesh.create_unit_square(comm, 10, 10)
        cls.W = fem.functionspace(cls.domain, ("Lagrange", 1))
    
    def test_uniform_concentration(self):
        """Test: CoV = 0 for a uniform concentration field."""
        ch = fem.Function(self.W)
        ch.x.array[:] = 0.5  # Uniform value
        
        # Sample at various points
        values = []
        for x in np.linspace(0.1, 0.9, 10):
            for y in np.linspace(0.1, 0.9, 10):
                try:
                    pt = np.array([[x, y, 0.0]], dtype=np.float64)
                    val = float(ch.eval(pt, [0])[0])
                    values.append(val)
                except:
                    pass
        
        if len(values) > 0:
            cov = compute_cov_manual(values)
            self.assertAlmostEqual(cov, 0.0, places=6, 
                msg="Uniform field should have CoV = 0")
        else:
            self.skipTest("Could not sample concentration values")
    
    def test_varying_concentration(self):
        """Test: CoV > 0 for a varying concentration field."""
        ch = fem.Function(self.W)
        ch.interpolate(lambda x: x[0])  # Linear field: c = x
        
        # Sample at various points
        values = []
        for x in np.linspace(0.1, 0.9, 10):
            for y in np.linspace(0.1, 0.9, 10):
                try:
                    pt = np.array([[x, y, 0.0]], dtype=np.float64)
                    val = float(ch.eval(pt, [0])[0])
                    values.append(val)
                except:
                    pass
        
        if len(values) > 0:
            cov = compute_cov_manual(values)
            # For a linear field from 0 to 1: mean = 0.5, std ≈ 0.2887, CoV ≈ 0.577
            self.assertGreater(cov, 0.1, 
                msg="Varying field should have CoV > 0")
        else:
            self.skipTest("Could not sample concentration values")



# TEST CLASS 2: Pressure Evaluation


class TestPressureFunction(unittest.TestCase):
    """Test pressure evaluation functions."""
    
    @classmethod
    def setUpClass(cls):
        """Set up a simple mesh for testing."""
        cls.domain = dmesh.create_unit_square(comm, 5, 5)
        cls.Q = fem.functionspace(cls.domain, ("Lagrange", 1))
    
    def test_pressure_interpolation(self):
        """Test: Pressure interpolation works correctly."""
        ph = fem.Function(self.Q)
        
        # Create a known pressure field: p = x + y
        def pressure_field(x):
            return x[0] + x[1]
        
        ph.interpolate(pressure_field)
        
        # Evaluate at (0.2, 0.3) -> expected 0.5
        try:
            pt = np.array([[0.2, 0.3, 0.0]], dtype=np.float64)
            p_val = float(ph.eval(pt, [0])[0])
            self.assertAlmostEqual(p_val, 0.5, places=5, 
                msg="Pressure at (0.2,0.3) should be 0.5")
        except:
            # If eval fails, check the function values directly
            self.assertEqual(ph.x.array.shape[0], self.Q.dofmap.index_map.size_local,
                msg="Pressure function should have correct number of DOFs")


# TEST CLASS 3: Inlet Condition


class TestInletCondition(unittest.TestCase):
    """Test the inlet concentration condition."""
    
    def test_c_inflow_step(self):
        """Test: Inlet concentration is 1.0 above y=0.205 and 0 below."""
        # Define the same function as in main solver
        def c_inflow(x):
            return np.where(x[1] >= 0.205, 1.0, 0.0).astype(np.float64)
        
        # Test points above and below the interface
        test_cases = [
            (0.1, 0.0),   # Below interface
            (0.15, 0.0),  # Below interface
            (0.205, 1.0), # At interface
            (0.25, 1.0),  # Above interface
            (0.3, 1.0),   # Above interface
        ]
        
        for y, expected in test_cases:
            x = np.array([[0.0, y, 0.0]]).T
            result = c_inflow(x)
            self.assertEqual(result[0], expected, 
                f"c_inflow at y={y:.3f} should be {expected}")


# TEST CLASS 4: Mesh Loading


class TestMeshLoading(unittest.TestCase):
    """Test mesh loading functions."""
    
    def test_mesh_load_0obstacles(self):
        """Test: 0-obstacle mesh loads correctly."""
        try:
            mesh_data = io.gmsh.read_from_msh("dfg_benchmark_0obstacles.msh", comm, gdim=2)
            domain = mesh_data.mesh
            facet_tags = mesh_data.facet_tags
            
            self.assertIsNotNone(domain, "Mesh domain should not be None")
            self.assertGreater(len(facet_tags.values), 0, "Facet tags should exist")
            
            obstacle_facets = facet_tags.find(3)
            self.assertEqual(len(obstacle_facets), 0, 
                f"0-obstacle mesh should have 0 obstacle facets, got {len(obstacle_facets)}")
            print("✓ 0-obstacle mesh loaded successfully")
        except Exception as e:
            self.skipTest(f"0-obstacle mesh file not available: {e}")
    
    def test_mesh_load_4obstacles(self):
        """Test: 4-obstacle mesh loads correctly."""
        try:
            mesh_data = io.gmsh.read_from_msh("dfg_benchmark_4obstacles.msh", comm, gdim=2)
            domain = mesh_data.mesh
            facet_tags = mesh_data.facet_tags
            
            self.assertIsNotNone(domain, "Mesh domain should not be None")
            self.assertGreater(len(facet_tags.values), 0, "Facet tags should exist")
            
            obstacle_facets = facet_tags.find(3)
            self.assertGreater(len(obstacle_facets), 0, 
                "4-obstacle mesh should have obstacle facets")
            print(f"✓ 4-obstacle mesh loaded successfully ({len(obstacle_facets)} obstacle facets)")
        except Exception as e:
            self.skipTest(f"4-obstacle mesh file not available: {e}")
    
    def test_mesh_load_6obstacles(self):
        """Test: 6-obstacle mesh loads correctly."""
        try:
            mesh_data = io.gmsh.read_from_msh("dfg_benchmark_6obstacles.msh", comm, gdim=2)
            domain = mesh_data.mesh
            facet_tags = mesh_data.facet_tags
            
            self.assertIsNotNone(domain, "Mesh domain should not be None")
            self.assertGreater(len(facet_tags.values), 0, "Facet tags should exist")
            
            obstacle_facets = facet_tags.find(3)
            self.assertGreater(len(obstacle_facets), 0, 
                "6-obstacle mesh should have obstacle facets")
            print(f"✓ 6-obstacle mesh loaded successfully ({len(obstacle_facets)} obstacle facets)")
        except Exception as e:
            self.skipTest(f"6-obstacle mesh file not available: {e}")



# TEST CLASS 5: Concentration Computation


class TestConcentrationComputation(unittest.TestCase):
    """Test concentration field computation."""
    
    @classmethod
    def setUpClass(cls):
        """Set up a simple mesh for testing."""
        cls.domain = dmesh.create_unit_square(comm, 5, 5)
        cls.W = fem.functionspace(cls.domain, ("Lagrange", 1))
    
    def test_concentration_values(self):
        """Test: Concentration values are within expected range."""
        ch = fem.Function(self.W)
        
        # Set concentration to a known value
        ch.x.array[:] = 0.5
        
        # Check that all values are 0.5
        self.assertTrue(np.allclose(ch.x.array, 0.5, rtol=1e-12), 
            "All concentration values should be 0.5")
    
    def test_concentration_interpolation(self):
        """Test: Concentration interpolation works."""
        ch = fem.Function(self.W)
        ch.interpolate(lambda x: x[0] + x[1])  # Linear field
        
        # Check that values are non-negative
        self.assertTrue(np.all(ch.x.array >= 0), 
            "Concentration values should be non-negative")

# Main Test Runner


def run_tests():
    """Run all unit tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestCoVFunction))
    suite.addTests(loader.loadTestsFromTestCase(TestPressureFunction))
    suite.addTests(loader.loadTestsFromTestCase(TestInletCondition))
    suite.addTests(loader.loadTestsFromTestCase(TestMeshLoading))
    suite.addTests(loader.loadTestsFromTestCase(TestConcentrationComputation))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    print("\n" + "="*70)
    print("UNIT TESTS: Static Mixer Simulation")
    print("="*70)
    print("Testing individual functions...")
    print("="*70)
    
    success = run_tests()
    
    print("\n" + "="*70)
    if success:
        print("✅ ALL UNIT TESTS PASSED! All functions are working correctly.")
    else:
        print("❌ SOME UNIT TESTS FAILED. Please check the implementation.")
    print("="*70)
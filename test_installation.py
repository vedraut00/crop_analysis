"""
Test Installation Script
Run this to verify all dependencies are installed correctly
"""

import sys

print("=" * 70)
print("TESTING INSTALLATION")
print("=" * 70)

# Test Python version
print(f"\nPython Version: {sys.version}")
if sys.version_info < (3, 7):
    print("❌ ERROR: Python 3.7 or higher required")
    sys.exit(1)
else:
    print("✓ Python version OK")

# Test imports
required_packages = {
    'numpy': 'NumPy',
    'pandas': 'Pandas',
    'matplotlib': 'Matplotlib',
    'seaborn': 'Seaborn',
    'sklearn': 'Scikit-learn'
}

print("\nTesting package imports...")
all_ok = True

for package, name in required_packages.items():
    try:
        __import__(package)
        print(f"✓ {name} installed")
    except ImportError:
        print(f"❌ {name} NOT installed")
        all_ok = False

if not all_ok:
    print("\n" + "=" * 70)
    print("INSTALLATION INCOMPLETE")
    print("=" * 70)
    print("\nPlease run:")
    print("  pip install -r requirements.txt")
    sys.exit(1)

# Test versions
print("\nPackage versions:")
try:
    import numpy as np
    print(f"  NumPy: {np.__version__}")
except:
    pass

try:
    import pandas as pd
    print(f"  Pandas: {pd.__version__}")
except:
    pass

try:
    import matplotlib
    print(f"  Matplotlib: {matplotlib.__version__}")
except:
    pass

try:
    import seaborn as sns
    print(f"  Seaborn: {sns.__version__}")
except:
    pass

try:
    import sklearn
    print(f"  Scikit-learn: {sklearn.__version__}")
except:
    pass

# Test basic functionality
print("\nTesting basic functionality...")

try:
    import numpy as np
    arr = np.array([1, 2, 3])
    assert arr.sum() == 6
    print("✓ NumPy working")
except Exception as e:
    print(f"❌ NumPy test failed: {e}")
    all_ok = False

try:
    import pandas as pd
    df = pd.DataFrame({'a': [1, 2, 3]})
    assert len(df) == 3
    print("✓ Pandas working")
except Exception as e:
    print(f"❌ Pandas test failed: {e}")
    all_ok = False

try:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    plt.close()
    print("✓ Matplotlib working")
except Exception as e:
    print(f"❌ Matplotlib test failed: {e}")
    all_ok = False

try:
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    print("✓ Scikit-learn working")
except Exception as e:
    print(f"❌ Scikit-learn test failed: {e}")
    all_ok = False

# Final result
print("\n" + "=" * 70)
if all_ok:
    print("✅ ALL TESTS PASSED")
    print("=" * 70)
    print("\nYou're ready to run the project!")
    print("\nNext steps:")
    print("  1. Run: python run_all.py")
    print("  2. Check generated files")
    print("  3. Review DOCUMENTATION.md")
    print("  4. Submit your work")
else:
    print("❌ SOME TESTS FAILED")
    print("=" * 70)
    print("\nPlease fix the issues above before running the project.")
    sys.exit(1)

print("\n" + "=" * 70)

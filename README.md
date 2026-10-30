# Daneel — Computational Astrophysics 2026–27

A teaching package for modelling exoplanet transits and building detection
pipelines. The repository contains the course example code, a runnable CLI,
and Sphinx documentation.

Documentation: <https://tiziano1590.github.io/comp_astro_26/>.

## Installation

### Prerequisites

- Python >= 3.10

### Install from source

```bash
git clone https://github.com/tiziano1590/comp_astro_26.git
cd comp_astro_26
pip install .
```

### Development installation

```bash
pip install -e '.[dev,docs]'
```

## Usage

After installation, you can run daneel from the command line:

```bash
daneel -i <input_file> --transit [--output lightcurve.png] [-s] [-e]
```
The package daneel requires an input parameter file (YAML format) containing the properties of the exoplanet, from these parameters it returns several outputs like transit, detection and atmosphere analysis.\
In the current version of the package only the detection of the exoplanet (transit) is implemented; stay tuned for future versions.

### Command-line options

- `-i, --input`: Input parameter file (required)
- `-t, --transit`: Generate a transit light curve from the YAML input
- `-d, --detect`: Reserved for the course detection exercise
- `-a, --atmosphere`: Reserved for the course atmosphere exercise
- `-o, --output`: Output path for the transit plot (default: `lc.png`)
- `-s, --show`: Display the light curve instead of only saving it
- `-e, --estimate`: Give a quick estimate of the flux at transit. Works only with --show

### Examples

```bash
# Run exoplanet detection
daneel -i examples/params.yaml --transit --output lightcurve.png

# Run exoplanet detection with displayed light curve
daneel -i examples/params.yaml --transit --output lightcurve.png --show

# Run exoplanet detection and give estimate of the flux at transit
daneel -i examples/params.yaml --transit -o lightcurve.png --show --estimate

# Run atmospheric characterization
daneel -i examples/params.yaml --detect

# Run both detection and atmospheric analysis
daneel -i examples/params.yaml --atmosphere
```

## Input File Format

The input file should be a YAML file containing the necessary parameters for the analysis.

## Development

Run the test suite and style checks with:

```bash
pytest
ruff check .
```

Build the documentation with `make -C docs html`.

## License

This project is licensed under the GNU General Public License v3.0.

## Author

Tiziano Zingales (tiziano.zingales@unipd.it)

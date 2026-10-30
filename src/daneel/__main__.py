import argparse
from datetime import datetime, timezone

from daneel.detection import TransitModel
from daneel.detection import TransitClass
from daneel.parameters import Parameters


def main():
    parser = argparse.ArgumentParser(description="Model and analyse exoplanet transits.")

    parser.add_argument(
        "-i",
        "--input",
        dest="input_file",
        type=str,
        required=True,
        help="Input par file to pass",
    )

    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("-t", "--transit", action="store_true", help="Plot a transit light curve")

    actions.add_argument("-d", "--detect", action="store_true", help="Run exoplanet detection")
    actions.add_argument("-a", "--atmosphere", action="store_true", help="Run atmospheric characterisation")
    parser.add_argument("-o", "--output", default="lc.png", help="Output image for --transit")
    parser.add_argument("-s","--show", default=False, action="store_true", help="Show the plot of the light curve. Hides computation time.")
    
    parser.add_argument("-e","--estimate", default=False, action="store_true", help="Quick estimate of transit depth, only if --show.") 

    args = parser.parse_args()

    """Launch Daneel"""
    if not args.show:
        start = datetime.now(timezone.utc)

    input_pars = Parameters(args.input_file).params

    if args.transit:
        transit = TransitClass(input_pars['transit'])
        if args.estimate and args.show:
            transit.get_transit_depth()
        transit.plot_transit(args.show, args.output)
    elif args.detect:
        parser.error("detection is not implemented yet; use --transit")
    elif args.atmosphere:
        parser.error("atmospheric characterisation is not implemented yet; use --transit")

    if not args.show:
        finish = datetime.now(timezone.utc)
        print(f"Daneel completed in {(finish - start).total_seconds():.2f}s")


if __name__ == "__main__":
    main()
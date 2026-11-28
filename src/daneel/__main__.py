import argparse
from datetime import datetime, timezone
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
    actions.add_argument("-d", "--detect", nargs='?', const=True, default=None, choices=['rf', 'cnn'], help="Run exoplanet detection (use '-d rf' or '--detection rf' for Random Forest and '-d cnn' for Convolutional Neural Network)")
    actions.add_argument("-a", "--atmosphere", action="store_true", help="Run atmospheric characterisation")
    actions.add_argument("--dream", action='store_true', help='Run Generative Adversarial Network and generate sample light curves')
    parser.add_argument("-o", "--output", default="lc.png", help="Output image for --transit")
    parser.add_argument("-s","--show", nargs='?', const=True, default=None, choices=['est'] , help="Show the plot of the light curve (use '-s est' to get an estimate of the depth of the transit). This command doesn't show computation time. ")
    
    args = parser.parse_args()

    """Launch Daneel"""
    if not args.show:
        start = datetime.now(timezone.utc)

    input_pars = Parameters(args.input_file).params

    if args.transit:
        from daneel.detection import TransitClass
        transit = TransitClass(input_pars['transit'])
        if args.show == 'est':
            transit.get_transit_depth()
        transit.plot_transit(args.show, args.output)
    elif args.detect == True:
        parser.error("detection is not implemented yet; use --transit")
    elif args.detect == "rf":
        from daneel.detection import RandomForestDetection
        rf = RandomForestDetection(input_pars)
        rf.main()
    elif args.detect == 'cnn':
        from daneel.detection import ConvolutionalNeuralNetwork
        cnn = ConvolutionalNeuralNetwork(input_pars)
        cnn.main()
        # print('ok con cnn')
    elif args.dream:
        from daneel.dream import GenerativeAdversarialNetwork
        gan = GenerativeAdversarialNetwork(input_pars)
        gan.train()
        gan.generate_and_plot(output_path="synthetic_light_curves.png")
    elif args.atmosphere:
        parser.error("atmospheric characterisation is not implemented yet; use --transit")

    if not args.show:
        finish = datetime.now(timezone.utc)
        print(f"Daneel completed in {(finish - start).total_seconds():.2f}s")


if __name__ == "__main__":
    main()
import argparse
import train_model
import run

def main():
    parser = argparse.ArgumentParser(description='Train and/or run the model')
    parser.add_argument('--mode', choices=['train', 'run', 'both'], required=True, help='Select mode: train, run or both')
    parser.add_argument('--start_date', type=str, required=True, help='Start date for training in format YYYY-MM-DD')
    parser.add_argument('--end_date', type=str, required=True, help='End date for training in format YYYY-MM-DD')

    args = parser.parse_args()

    if args.mode == 'train' or args.mode == 'both':
        train_model.train(args.start_date, args.end_date)

    if args.mode == 'run' or args.mode == 'both':
        run.run_model()

if __name__ == '__main__':
    main()

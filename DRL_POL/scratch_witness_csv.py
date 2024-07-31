import csv


def read_witness_dat(file_path):
    points = []
    with open(file_path, "r") as file:
        lines = file.readlines()
        for line in lines:
            if line.startswith("WITNESS_POINTS") or line.startswith(
                "END_WITNESS_POINTS"
            ):
                continue
            x, y, z = map(float, line.strip().split(","))
            points.append((x, y, z))
    return points


def filter_points(points, x_range, z_range):
    filtered_points = [
        point
        for point in points
        if x_range[0] < point[0] < x_range[1] and z_range[0] < point[2] < z_range[1]
    ]
    return filtered_points


def save_to_csv(points, output_file):
    with open(output_file, "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["x", "y", "z"])
        csv_writer.writerows(points)


def main():
    input_file = "witness.dat"
    output_file = "filtered_witness_points_0-1.csv"

    x_range = (0.0, 0.5)
    z_range = (0.5, 1)

    points = read_witness_dat(input_file)
    filtered_points = filter_points(points, x_range, z_range)
    # Multiply the x and z coordinates by 2.67 and 0,8 respecitvely
    filtered_points = [(2.67 * x, 1.8 * y, 0.8 * z) for x, y, z in filtered_points]
    save_to_csv(filtered_points, output_file)
    print("Done!")


if __name__ == "__main__":
    main()

package Java;

import java.util.ArrayList;

public class Ballistics {
    static double gravity = 9.81; // m/s^2
    static Vector3d blueRimPosition = new Vector3d(0, 0, 0); // need to add
    static Vector3d redRimPosition = new Vector3d(0, 0, 0); // need to add

    public static Vector3d calculatePositionAtTime(double time, boolean isBlueRim, double timeAtRim, ArrayList<PositionAtTime> positionsList){
        double vX = calculateVx(positionsList);
        double vY = calculateVy(positionsList);
        Vector3d initialPosition = calculateInitialPosition(positionsList, timeAtRim, isBlueRim);
        double x = initialPosition.x() + vX * time;
        double y = initialPosition.y() + vY * time;
        double z = initialPosition.z() + (0.5 * gravity * time * time);

        return new Vector3d(x, y, z);
    }




    
    public static Vector3d calculateInitialPosition(ArrayList<PositionAtTime> positions, double timeAtRim,
            boolean isBlueRim) {
        if (positions.isEmpty()) {
            return new Vector3d(0, 0, 0);
        }

        double vX = calculateVx(positions); // constant during flight
        double vY = calculateVy(positions); // constant during flight
        Vector3d rimPosition = isBlueRim ? blueRimPosition : redRimPosition;
        double initialX = rimPosition.x() - (vX * timeAtRim);
        double initialY = rimPosition.y() - (vY * timeAtRim);
        double initialZ = calculateInitialZ(positions);


        return new Vector3d(initialX, initialY, initialZ);
    }





    public static double calculateInitialZ(ArrayList<PositionAtTime> positions) {
        int n = positions.size();
        double sumT = 0, sumZprime = 0, sumTT = 0, sumTZprime = 0;
        for (int i = 0; i < n; i++) {
            double t = positions.get(i).t();
            double zPrime = positions.get(i).z() + 0.5 * gravity * t * t;
            sumT += t;
            sumZprime += zPrime;
            sumTT += t * t;
            sumTZprime += t * zPrime;
        }

        double meanT = sumT / n;
        double meanZprime = sumZprime / n;

        double numerator = sumTZprime - n * meanT * meanZprime;
        double denominator = sumTT - n * meanT * meanT;
        double vz0 = numerator / denominator;
        double z0 = meanZprime - vz0 * meanT;

        return z0;

    }

    public static double calculateVx(ArrayList<PositionAtTime> positions) {
        double vX = 0;
        for (int i = 0; i < positions.size() - 1; i++) {
            double deltaX = positions.get(i + 1).x() - positions.get(i).x();
            double deltaT = positions.get(i + 1).t() - positions.get(i).t();
            vX += (deltaX / deltaT) * (1 / positions.size());
        }
        return vX;

    }

    public static double calculateVy(ArrayList<PositionAtTime> positions) {
        double vY = 0;
        for (int i = 0; i < positions.size() - 1; i++) {
            double deltaY = positions.get(i + 1).y() - positions.get(i).y();
            double deltaT = positions.get(i + 1).t() - positions.get(i).t();
            vY += (deltaY / deltaT) * (1 / positions.size());
        }
        return vY;

    }
}

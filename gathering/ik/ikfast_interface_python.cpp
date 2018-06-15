/*
 * IKFast Demo
 * 
 * Shows how to calculate FK from joint angles.
 * Calculates IK from rotation-translation matrix, or translation-quaternion pose.
 * Performance timing tests.
 *
 * Run the program to view command line parameters.
 * 
 * 
 * To compile, run:
 * g++ -lstdc++ -llapack -o compute ikfastdemo.cpp -lrt
 * (need to link with 'rt' for gettime(), it must come after the source file name)
 *
 * 
 * Tested with Ubuntu 11.10 (Oneiric)
 * IKFast54 from OpenRAVE 0.6.0
 * IKFast56/61 from OpenRave 0.8.2
 *
 * Author: David Butterworth, KAIST
 *         Based on code by Rosen Diankov
 * Date: November 2012
 */

/*
 * Copyright (c) 2012, David Butterworth, KAIST
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the Willow Garage, Inc. nor the names of its
 *       contributors may be used to endorse or promote products derived from
 *       this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#define IKFAST_HAS_LIBRARY // Build IKFast with API functions
#define IKFAST_NO_MAIN // Don't include main() from IKFast

/*
Set which IKFast version you are using
The API calls are slightly different for versions > 54
*/

#define IK_VERSION 61
#include "ikfast61_irb1600id.cpp"


//----------------------------------------------------------------------------//

#include <stdio.h>
#include <stdlib.h>
#include <time.h> // for clock_gettime()
#include <math.h>

float SIGN(float x);
float NORM(float a, float b, float c, float d);

#define IKREAL_TYPE IkReal // for IKFast 56,61

const double joint_limits[6][2] = {
    {-2.35619449019, 2.35619449019},
    {-1.57079632679, 2.61799387799},
    {-1.57079632679, 1.37881010908},
    {-2.70526034059, 2.70526034059},
    {-1.57079632679, 2.35619449019},
    {-4.799655442984406, 4.799655442984406}};


bool in_joint_range(double q[]){
    for(int i=0; i<6; i++){
        if(q[i] < joint_limits[i][0] || q[i] > joint_limits[i][1]){
            return false;
        }
    }
    return true;
}

double cost(double q[], double q0[], double weight[]){
    double sum = 0;
    double tmp;
    for(int i=0; i<6; i++){
        tmp = (q[i] - q0[i]) * weight[i];
        sum += tmp*tmp;
    }
    return sum;
}

void _copy(double x[], double y[], const int size = 6){
    for(int i=0;i<size;i++)
        x[i] = y[i];
}

int select_best_js(double best_q[], double candidates[], int ncandidate, double q0[], double weight[]){
    // we prefer minimum travel
    double min_v = 1e9;  // some large number
    double v;
    int ret = 0;  // 0: not found, 1: found
    for(int i=0; i<ncandidate; i++){
        double* q = &candidates[i*6];
        v = cost( q , q0 , weight);
        if(v < min_v && in_joint_range(q)){
            min_v = v;
            _copy(best_q, q);
            ret = 1;
        }
    }
    return ret;
}

float SIGN(float x) {
    return (x >= 0.0f) ? +1.0f : -1.0f;
}

float NORM(float a, float b, float c, float d) {
    return sqrt(a * a + b * b + c * c + d * d);
}

    
extern "C" {
// soltionList will be an 1d array consists of nsol*6 numbers
// solutionList should be of size at least 6*5
void ikfastPython(double* solutionList, int* nsol, double* pose){
    unsigned int num_of_joints = GetNumJoints();
    IKREAL_TYPE eerot[9],eetrans[3];
    
    IkSolutionList<IKREAL_TYPE> solutions;
    
    eetrans[0] = (pose[0]);
    eetrans[1] = (pose[1]);
    eetrans[2] = (pose[2]);

    // Convert input effector pose, in w x y z quaternion notation, to rotation matrix. 
    // Must use doubles, else lose precision compared to directly inputting the rotation matrix.
    double qw = (pose[3]);
    double qx = (pose[4]);
    double qy = (pose[5]);
    double qz = (pose[6]);
    const double n = 1.0f/sqrt(qx*qx+qy*qy+qz*qz+qw*qw);
    qw *= n;
    qx *= n;
    qy *= n;
    qz *= n;
    eerot[0] = 1.0f - 2.0f*qy*qy - 2.0f*qz*qz;  eerot[1] = 2.0f*qx*qy - 2.0f*qz*qw;         eerot[2] = 2.0f*qx*qz + 2.0f*qy*qw;
    eerot[3] = 2.0f*qx*qy + 2.0f*qz*qw;         eerot[4] = 1.0f - 2.0f*qx*qx - 2.0f*qz*qz;  eerot[5] = 2.0f*qy*qz - 2.0f*qx*qw;
    eerot[6] = 2.0f*qx*qz - 2.0f*qy*qw;         eerot[7] = 2.0f*qy*qz + 2.0f*qx*qw;         eerot[8] = 1.0f - 2.0f*qx*qx - 2.0f*qy*qy;
    
    bool bSuccess = ComputeIk(eetrans, eerot, NULL, solutions);
    if( !bSuccess ) {
        *nsol = 0;
        return;
    }
    
    // output the result
    unsigned int num_of_solutions = (int)solutions.GetNumSolutions();

    std::vector<IKREAL_TYPE> solvalues(num_of_joints);
    for(std::size_t i = 0; i < num_of_solutions; ++i) {
        const IkSolutionBase<IKREAL_TYPE>& sol = solutions.GetSolution(i);
        int this_sol_free_params = (int)sol.GetFree().size(); 
        std::vector<IKREAL_TYPE> vsolfree(this_sol_free_params);

        sol.GetSolution(&solvalues[0],vsolfree.size()>0?&vsolfree[0]:NULL);
        
        *nsol = num_of_solutions;
        
        for( std::size_t j = 0; j < solvalues.size(); ++j)
            solutionList[i*6 + j]= solvalues[j];
    }
}

// solution: 6 numbers representing joint positions
// pose: the target pose
// weight: 6 numbers to weigh the difference in 6 joint motion
// q0: 6 numbers represeting the nominal joint
void ikfastAndFindBest(double* solution, double* pose, double* weight, double* q0, int* hassol){
    const int solutionListSize = 20*3;
    double solutionList[6*solutionListSize];
    int nsol, nsol_more;  
    
    ikfastPython(solutionList, &nsol, pose);
    //add and subtract at joint 6 to obtain more possible solutions
    nsol_more = nsol;
    for(int i=0; i<nsol; i++){
        _copy( & solutionList[nsol_more * 6] , & solutionList[ i*6 ] );
        solutionList[nsol_more * 6 + 5] += 2 * M_PI;
        nsol_more++;
        if(nsol_more >= solutionListSize) break;  // just to be careful
        
        _copy( & solutionList[nsol_more * 6] , & solutionList[ i*6 ] );
        solutionList[nsol_more * 6 + 5] -= 2 * M_PI;
        nsol_more++;
        if(nsol_more >= solutionListSize) break;
    }
    
    *hassol = select_best_js(solution, solutionList, nsol_more, q0, weight);
}

// return pose = x,y,z,qx,qy,qz,qw, q0 = [q1,q2,q3,q4,q5,q6] in rad
void fkfastPython(double* pose, double* joints){
    IKREAL_TYPE eerot[9],eetrans[3];
    
    ComputeFk(joints, eetrans, eerot);
        
    float q0 = ( eerot[0] + eerot[4] + eerot[8] + 1.0f) / 4.0f;
    float q1 = ( eerot[0] - eerot[4] - eerot[8] + 1.0f) / 4.0f;
    float q2 = (-eerot[0] + eerot[4] - eerot[8] + 1.0f) / 4.0f;
    float q3 = (-eerot[0] - eerot[4] + eerot[8] + 1.0f) / 4.0f;
    if(q0 < 0.0f) q0 = 0.0f;
    if(q1 < 0.0f) q1 = 0.0f;
    if(q2 < 0.0f) q2 = 0.0f;
    if(q3 < 0.0f) q3 = 0.0f;
    q0 = sqrt(q0);
    q1 = sqrt(q1);
    q2 = sqrt(q2);
    q3 = sqrt(q3);
    if(q0 >= q1 && q0 >= q2 && q0 >= q3) {
        q0 *= +1.0f;
        q1 *= SIGN(eerot[7] - eerot[5]);
        q2 *= SIGN(eerot[2] - eerot[6]);
        q3 *= SIGN(eerot[3] - eerot[1]);
    } else if(q1 >= q0 && q1 >= q2 && q1 >= q3) {
        q0 *= SIGN(eerot[7] - eerot[5]);
        q1 *= +1.0f;
        q2 *= SIGN(eerot[3] + eerot[1]);
        q3 *= SIGN(eerot[2] + eerot[6]);
    } else if(q2 >= q0 && q2 >= q1 && q2 >= q3) {
        q0 *= SIGN(eerot[2] - eerot[6]);
        q1 *= SIGN(eerot[3] + eerot[1]);
        q2 *= +1.0f;
        q3 *= SIGN(eerot[7] + eerot[5]);
    } else if(q3 >= q0 && q3 >= q1 && q3 >= q2) {
        q0 *= SIGN(eerot[3] - eerot[1]);
        q1 *= SIGN(eerot[6] + eerot[2]);
        q2 *= SIGN(eerot[7] + eerot[5]);
        q3 *= +1.0f;
    } else {
        printf("Error while converting to quaternion! \n");
    }
    float r = NORM(q0, q1, q2, q3);
    q0 /= r;
    q1 /= r;
    q2 /= r;
    q3 /= r;
    pose[0] = eetrans[0];
    pose[1] = eetrans[1];
    pose[2] = eetrans[2];
    pose[3] = q0;
    pose[4] = q1;
    pose[5] = q2;
    pose[6] = q3;
}
}


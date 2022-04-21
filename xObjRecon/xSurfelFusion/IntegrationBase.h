//
//  integration_base.h
//  VINS_ios
//
//  Created by HKUST Aerial Robotics on 2016/11/25.
//  Copyright © 2017 HKUST Aerial Robotics. All rights reserved.
//

#ifndef integration_base_h
#define integration_base_h

#include <Eigen/Eigen>

#include "xUtils.h"

#define ACC_N ((double)0.5)  //0.02
#define ACC_W ((double)0.002)
#define GYR_N ((double)0.2)  //0.02
#define GYR_W ((double)4.0e-5)

enum StateOrder
{
	O_P = 0,
	O_R = 3,
	O_V = 6,
	O_BA = 9,
	O_BG = 12
};

//#include "utility.hpp"
//#include <ceres/ceres.h>
//#include "global_param.hpp"

using namespace Eigen;
class IntegrationBase
{
public:
    IntegrationBase()
    {
        noise = Eigen::Matrix<double, 18, 18>::Zero();
        noise.block<3, 3>(0, 0) =  (ACC_N * ACC_N) * Eigen::Matrix3d::Identity();
        noise.block<3, 3>(3, 3) =  (GYR_N * GYR_N) * Eigen::Matrix3d::Identity();
        noise.block<3, 3>(6, 6) =  (ACC_N * ACC_N) * Eigen::Matrix3d::Identity();
        noise.block<3, 3>(9, 9) =  (GYR_N * GYR_N) * Eigen::Matrix3d::Identity();
        noise.block<3, 3>(12, 12) =  (ACC_W * ACC_W) * Eigen::Matrix3d::Identity();
		noise.block<3, 3>(15, 15) =  (GYR_W * GYR_W) * Eigen::Matrix3d::Identity();
    }
    
    void repropagate(const Eigen::Vector3d &_linearized_ba, const Eigen::Vector3d &_linearized_bg)
    {
#if 0
        sum_dt = 0.0;
        acc_0 = linearized_acc;
        gyr_0 = linearized_gyr;
        delta_p.setZero();
        delta_q.setIdentity();
        delta_v.setZero();
        linearized_ba = _linearized_ba;
        linearized_bg = _linearized_bg;
        jacobian.setIdentity();
        covariance.setZero();
        for (int i = 0; i < static_cast<int>(dt_buf.size()); i++)
            propagate(dt_buf[i], acc_buf[i], gyr_buf[i]);
#endif
    }
    
    void midPointIntegration(double _dt,
                             const Eigen::Vector3d &_acc_0, const Eigen::Vector3d &_gyr_0,
                             const Eigen::Vector3d &_acc_1, const Eigen::Vector3d &_gyr_1,
                             const Eigen::Vector3d &delta_p, const Eigen::Quaterniond &delta_q, const Eigen::Vector3d &delta_v,
                             const Eigen::Vector3d &linearized_ba, const Eigen::Vector3d &linearized_bg,
                             Eigen::Vector3d &result_delta_p, Eigen::Quaterniond &result_delta_q, Eigen::Vector3d &result_delta_v,
                             Eigen::Vector3d &result_linearized_ba, Eigen::Vector3d &result_linearized_bg, bool update_jacobian)
    {
        Vector3d un_acc_0 = delta_q * (_acc_0 - linearized_ba);
        Vector3d un_gyr = 0.5 * (_gyr_0 + _gyr_1) - linearized_bg;
        result_delta_q = delta_q * Quaterniond(1, un_gyr(0) * _dt / 2, un_gyr(1) * _dt / 2, un_gyr(2) * _dt / 2);
        Vector3d un_acc_1 = result_delta_q * (_acc_1 - linearized_ba);
        Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
        result_delta_p = delta_p + delta_v * _dt + 0.5 * un_acc * _dt * _dt;
        result_delta_v = delta_v + un_acc * _dt;
        result_linearized_ba = linearized_ba;
        result_linearized_bg = linearized_bg;
        
        if(update_jacobian)
        {
            Vector3d w_x = 0.5 * (_gyr_0 + _gyr_1) - linearized_bg;
            Vector3d a_0_x = _acc_0 - linearized_ba;
            Vector3d a_1_x = _acc_1 - linearized_ba;
            Matrix3d R_w_x, R_a_0_x, R_a_1_x;
            
            R_w_x<< 0, -w_x(2), w_x(1),
                    w_x(2), 0, -w_x(0),
                    -w_x(1), w_x(0), 0;
            
            R_a_0_x<< 0, -a_0_x(2), a_0_x(1),
                      a_0_x(2), 0, -a_0_x(0),
                      -a_0_x(1), a_0_x(0), 0;
            
            R_a_1_x<< 0, -a_1_x(2), a_1_x(1),
                      a_1_x(2), 0, -a_1_x(0),
                      -a_1_x(1), a_1_x(0), 0;
            
            MatrixXd F = MatrixXd::Zero(15, 15);
            F.block<3, 3>(0, 0) = Matrix3d::Identity();
            F.block<3, 3>(0, 3) = -0.25 * delta_q.toRotationMatrix() * R_a_0_x * _dt * _dt +
            -0.25 * result_delta_q.toRotationMatrix() * R_a_1_x * (Matrix3d::Identity() - R_w_x * _dt) * _dt * _dt;
            F.block<3, 3>(0, 6) = MatrixXd::Identity(3,3) * _dt;
            F.block<3, 3>(0, 9) = -0.25 * (delta_q.toRotationMatrix() + result_delta_q.toRotationMatrix()) * _dt * _dt;
            F.block<3, 3>(0, 12) = -0.25 * result_delta_q.toRotationMatrix() * R_a_1_x * _dt * _dt * -_dt;
            F.block<3, 3>(3, 3) = Matrix3d::Identity() - R_w_x * _dt;
            F.block<3, 3>(3, 12) = -1.0 * MatrixXd::Identity(3,3) * _dt;
            F.block<3, 3>(6, 3) = -0.5 * delta_q.toRotationMatrix() * R_a_0_x * _dt +
            -0.5 * result_delta_q.toRotationMatrix() * R_a_1_x * (Matrix3d::Identity() - R_w_x * _dt) * _dt;
            F.block<3, 3>(6, 6) = Matrix3d::Identity();
            F.block<3, 3>(6, 9) = -0.5 * (delta_q.toRotationMatrix() + result_delta_q.toRotationMatrix()) * _dt;
            F.block<3, 3>(6, 12) = -0.5 * result_delta_q.toRotationMatrix() * R_a_1_x * _dt * -_dt;
            F.block<3, 3>(9, 9) = Matrix3d::Identity();
            F.block<3, 3>(12, 12) = Matrix3d::Identity();
            //cout<<"A"<<endl<<A<<endl;
            
#if 0
            MatrixXd V = MatrixXd::Zero(15,18);
            V.block<3, 3>(0, 0) =  0.25 * delta_q.toRotationMatrix() * _dt * _dt;
            V.block<3, 3>(0, 3) =  0.25 * -result_delta_q.toRotationMatrix() * R_a_1_x  * _dt * _dt * 0.5 * _dt;
            V.block<3, 3>(0, 6) =  0.25 * result_delta_q.toRotationMatrix() * _dt * _dt;
            V.block<3, 3>(0, 9) =  V.block<3, 3>(0, 3);
            V.block<3, 3>(3, 3) =  0.5 * MatrixXd::Identity(3,3) * _dt;
            V.block<3, 3>(3, 9) =  0.5 * MatrixXd::Identity(3,3) * _dt;
            V.block<3, 3>(6, 0) =  0.5 * delta_q.toRotationMatrix() * _dt;
            V.block<3, 3>(6, 3) =  0.5 * -result_delta_q.toRotationMatrix() * R_a_1_x  * _dt * 0.5 * _dt;
            V.block<3, 3>(6, 6) =  0.5 * result_delta_q.toRotationMatrix() * _dt;
            V.block<3, 3>(6, 9) =  V.block<3, 3>(6, 3);
            V.block<3, 3>(9, 12) = MatrixXd::Identity(3,3) * _dt;
            V.block<3, 3>(12, 15) = MatrixXd::Identity(3,3) * _dt;
#endif
            
            //step_jacobian = F;
            //step_V = V;
            jacobian = F * jacobian;
#if 0
            covariance = F * covariance * F.transpose() + V * noise * V.transpose();
#endif
        }
    }
    
    void propagate(double dt, const double3& acc_1, const double3& gyr_1)
    {
        m_dt = dt;
		m_acc_1 = Eigen::Vector3d({ acc_1.x, acc_1.y, acc_1.z });
        m_gyr_1 = Eigen::Vector3d({ gyr_1.x, gyr_1.y, gyr_1.z });

        midPointIntegration(m_dt, m_acc_0, m_gyr_0, m_acc_1, m_gyr_1, m_delta_p, m_delta_q, m_delta_v,
                            m_linearized_ba, m_linearized_bg,
                            m_result_delta_p, m_result_delta_q, m_result_delta_v,
                            m_result_linearized_ba, m_result_linearized_bg, 1);
        
        m_delta_p = m_result_delta_p;
        m_delta_q = m_result_delta_q;
        m_delta_q.normalize();
        m_delta_v = m_result_delta_v;

        m_linearized_ba = m_result_linearized_ba;
        m_linearized_bg = m_result_linearized_bg;

        m_sum_dt += dt;

        m_acc_0 = m_acc_1;
        m_gyr_0 = m_gyr_1;
    }
    
#if 0
    Eigen::Matrix<double, 15, 1> calcResidual()
    {
        Eigen::Matrix<double, 15, 1> residuals;
  
        Eigen::Matrix3d dp_dba = jacobian.block<3, 3>(O_P, O_BA);
        Eigen::Matrix3d dp_dbg = jacobian.block<3, 3>(O_P, O_BG);
        
        Eigen::Matrix3d dq_dbg = jacobian.block<3, 3>(O_R, O_BG);
        
        Eigen::Matrix3d dv_dba = jacobian.block<3, 3>(O_V, O_BA);
        Eigen::Matrix3d dv_dbg = jacobian.block<3, 3>(O_V, O_BG);
        
        Eigen::Vector3d dba = Bai - linearized_ba;
        Eigen::Vector3d dbg = Bgi - linearized_bg;
        
        Eigen::Quaterniond corrected_delta_q = delta_q * Utility::deltaQ(dq_dbg * dbg);
        Eigen::Vector3d corrected_delta_v = delta_v + dv_dba * dba + dv_dbg * dbg;
        Eigen::Vector3d corrected_delta_p = delta_p + dp_dba * dba + dp_dbg * dbg;
        
        Vector3d G{0,0,GRAVITY};
        residuals.block<3, 1>(O_P, 0) = Qi.inverse() * (0.5 * G * sum_dt * sum_dt + Pj - Pi - Vi * sum_dt) - corrected_delta_p;
        residuals.block<3, 1>(O_R, 0) = 2 * (corrected_delta_q.inverse() * (Qi.inverse() * Qj)).vec();
        residuals.block<3, 1>(O_V, 0) = Qi.inverse() * (G * sum_dt + Vj - Vi) - corrected_delta_v;
        residuals.block<3, 1>(O_BA, 0) = Baj - Bai;
        residuals.block<3, 1>(O_BG, 0) = Bgj - Bgi;
        return residuals;
    }
#endif

	Eigen::Matrix3d CrossMat(Eigen::Vector3d &vec)
    {
		Eigen::Matrix3d res = Eigen::Matrix3d::Zero();
		res(0, 1) = vec.z();
		res(0, 2) = -vec.y();
		res(1, 0) = -vec.z();
		res(1, 2) = vec.x();
		res(2, 0) = vec.y();
		res(2, 1) = -vec.x();
		
		return res;
    }

	void imuStep(Eigen::Matrix<double, 4, 4, Eigen::RowMajor>& relativeRt,
	             Eigen::Matrix<float, 3, 3, Eigen::RowMajor>& RPrevInv,
	             Eigen::Vector3f& tPrev,
	             Eigen::Vector3f& velocityPrev,
	             Eigen::Vector3f& biasAccPrev,
	             Eigen::Vector3f& biasGyrPrev,
	             ImuMeasurements& imuMeasurements,
	             Gravity& gravityW,
	             Eigen::Matrix<double, 15, 15, Eigen::RowMajor>& A_imu,
	             Eigen::Matrix<double, 15, 1>& b_imu)
    {
		m_sum_dt = 0.0;	
		m_delta_p.setZero();
		m_delta_q.setIdentity();
		m_delta_v.setZero();
		m_linearized_ba = biasAccPrev.cast<double>();
		m_linearized_bg = biasGyrPrev.cast<double>();
		jacobian.setIdentity();
		covariance.setZero();

		double current_time = -1.0;
		for (int i = 0; i < imuMeasurements.size(); ++i)
		{
			ImuMsg &imuMsg = imuMeasurements[i];
			double t = imuMsg.timeStamp;
			if (current_time < 0)
				current_time = t;
			double dt = (t - current_time);
			current_time = t;

			if (i == 0)
			{
				m_acc_0 = Eigen::Vector3d({ imuMsg.acc.x, imuMsg.acc.y, imuMsg.acc.z });;
				m_gyr_0 = Eigen::Vector3d({ imuMsg.gyr.x, imuMsg.gyr.y, imuMsg.gyr.z });;
			}
			propagate(dt, imuMsg.acc, imuMsg.gyr);
		}

		Eigen::Vector3d relativet = relativeRt.topRightCorner(3, 1);
		Eigen::Matrix<double, 3, 3, Eigen::RowMajor> relativeR = relativeRt.topLeftCorner(3, 3);
		step_residual.setZero();
		step_jacobian.setZero();

		//std::cout << "relativet: " << relativet << std::endl;
		//std::cout << "m_delta_p: " << m_delta_p << std::endl;
		step_residual.block<3, 1>(0, 0) = relativet -
			RPrevInv.cast<double>() * velocityPrev.cast<double>() * m_sum_dt -
			m_delta_p;

		Eigen::AngleAxisd angleAxisRelativeRDeltaR;
		angleAxisRelativeRDeltaR.fromRotationMatrix(relativeR * m_delta_q.toRotationMatrix().inverse());
		step_residual.block<3, 1>(3, 0) = -angleAxisRelativeRDeltaR.axis() * angleAxisRelativeRDeltaR.angle();

		step_jacobian.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
		step_jacobian.block<3, 3>(0, 3) = CrossMat(relativet);
		step_jacobian.block<3, 3>(0, 6) = -RPrevInv.cast<double>() * m_sum_dt;
		step_jacobian.block<3, 3>(0, 9) = jacobian.block<3, 3>(0, 9);
		step_jacobian.block<3, 3>(0, 12) = jacobian.block<3, 3>(0, 12);

		step_jacobian.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity();
		step_jacobian.block<3, 3>(3, 9) = jacobian.block<3, 3>(3, 9);
		step_jacobian.block<3, 3>(3, 12) = jacobian.block<3, 3>(3, 12);

#if 0
		std::cout << jacobian.block<3, 3>(0, 9) << std::endl;
		std::cout << jacobian.block<3, 3>(0, 12) << std::endl;
		std::cout << jacobian.block<3, 3>(3, 9) << std::endl;
		std::cout << jacobian.block<3, 3>(3, 12) << std::endl;
		std::exit(0);
#endif

		A_imu = step_jacobian.transpose() * step_jacobian;
		b_imu = -step_jacobian.transpose() * step_residual;

#if 0
		Eigen::Vector3d relativet = relativeRt.topRightCorner(3, 1);
		Eigen::Matrix<double, 3, 3, Eigen::RowMajor> relativeR = relativeRt.topLeftCorner(3, 3);

		step_residual.block<3, 1>(O_P, 0) = relativet -
			RPrevInv.cast<double>() * velocityCurr.cast<double>() * m_sum_dt -
			m_delta_p;

		Eigen::AngleAxisd angleAxisRelativeRDeltaR;
		angleAxisRelativeRDeltaR.fromRotationMatrix(relativeR * m_delta_q.toRotationMatrix().inverse());
		step_residual.block<3, 1>(O_R, 0) = -angleAxisRelativeRDeltaR.axis() * angleAxisRelativeRDeltaR.angle();

		step_residual.block<3, 1>(O_V, 0) = RPrevInv.cast<double>() *
			(velocityCurr.cast<double>() - velocityPrev.cast<double>()) - m_delta_v;

		step_residual.block<3, 1>(O_BA, 0) = biasAccCurr.cast<double>() - biasAccPrev.cast<double>();

		step_residual.block<3, 1>(O_BG, 0) = biasGyrCurr.cast<double>() - biasGyrPrev.cast<double>();

		step_jacobian.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
		step_jacobian.block<3, 3>(0, 3) = CrossMat(relativet);
		step_jacobian.block<3, 3>(0, 6) = -RPrevInv.cast<double>() * m_sum_dt;
		step_jacobian.block<3, 3>(0, 9) = jacobian.block<3, 3>(0, 9);
		step_jacobian.block<3, 3>(0, 12) = jacobian.block<3, 3>(0, 12);

		step_jacobian.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity();
		step_jacobian.block<3, 3>(3, 9) = jacobian.block<3, 3>(3, 9);
		step_jacobian.block<3, 3>(3, 12) = jacobian.block<3, 3>(9, 12);

		step_jacobian.block<3, 3>(6, 6) = -RPrevInv.cast<double>();
		step_jacobian.block<3, 3>(6, 9) = jacobian.block<3, 3>(6, 9);
		step_jacobian.block<3, 3>(6, 12) = jacobian.block<3, 3>(6, 12);

		step_jacobian.block<3, 3>(9, 9) = Eigen::Matrix3d::Identity();

		step_jacobian.block<3, 3>(12, 12) = Eigen::Matrix3d::Identity();

		A_imu = step_jacobian.transpose() * step_jacobian;
		b_imu = -step_jacobian.transpose() * step_residual;
#endif

#if 0
		Eigen::Vector3d relativet = relativeRt.topRightCorner(3, 1);
		Eigen::Matrix<double, 3, 3, Eigen::RowMajor> relativeR = relativeRt.topLeftCorner(3, 3);

		step_residual.block<3, 1>(O_P, 0) = relativet +
		    RPrevInv.cast<double>() * (0.5 * gravityW.cast<double>() * m_sum_dt * m_sum_dt - velocityCurr.cast<double>() * m_sum_dt) -
		    m_delta_p;
		
		Eigen::AngleAxisd angleAxisRelativeRDeltaR;
		angleAxisRelativeRDeltaR.fromRotationMatrix(relativeR * m_delta_q.toRotationMatrix().inverse());
		step_residual.block<3, 1>(O_R, 0) = -angleAxisRelativeRDeltaR.axis() * angleAxisRelativeRDeltaR.angle();

		step_residual.block<3, 1>(O_V, 0) = RPrevInv.cast<double>() * 
			(gravityW.cast<double>() * m_sum_dt + velocityCurr.cast<double>() - velocityPrev.cast<double>()) - m_delta_v;

		step_residual.block<3, 1>(O_BA, 0) = biasAccCurr.cast<double>() - biasAccPrev.cast<double>();

		step_residual.block<3, 1>(O_BG, 0) = biasGyrCurr.cast<double>() - biasGyrPrev.cast<double>();

		step_jacobian.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
		step_jacobian.block<3, 3>(0, 3) = CrossMat(relativet);

		step_jacobian.block<3, 3>(0, 9) = jacobian.block<3, 3>(0, 9);
		step_jacobian.block<3, 3>(0, 12) = jacobian.block<3, 3>(0, 12);

		step_jacobian.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity();
		step_jacobian.block<3, 3>(3, 9) = jacobian.block<3, 3>(3, 9);
		step_jacobian.block<3, 3>(3, 12) = jacobian.block<3, 3>(9, 12);

		step_jacobian.block<3, 3>(6, 6) = RPrevInv.cast<double>();
		step_jacobian.block<3, 3>(6, 9) = jacobian.block<3, 3>(6, 9);
		step_jacobian.block<3, 3>(6, 12) = jacobian.block<3, 3>(6, 12);

		step_jacobian.block<3, 3>(9, 9) = Eigen::Matrix3d::Identity();

		step_jacobian.block<3, 3>(12, 12) = Eigen::Matrix3d::Identity();

		A_imu = step_jacobian.transpose() * step_jacobian;
		b_imu = - step_jacobian.transpose() * step_residual;
#endif
    }
    
    double m_dt;
    Eigen::Vector3d m_acc_0, m_gyr_0;
    Eigen::Vector3d m_acc_1, m_gyr_1;

	double m_sum_dt;
	Eigen::Vector3d m_delta_p, m_result_delta_p;
	Eigen::Quaterniond m_delta_q, m_result_delta_q;
	Eigen::Vector3d m_delta_v, m_result_delta_v;
	Vector3d m_linearized_ba, m_result_linearized_ba;
	Vector3d m_linearized_bg, m_result_linearized_bg;
    
    Eigen::Matrix<double, 15, 15> jacobian, covariance;
    Eigen::Matrix<double, 15, 15> step_jacobian;
    Eigen::Matrix<double, 15, 1> step_residual;
    Eigen::Matrix<double, 18, 18> noise;
    
    std::vector<double> dt_buf;
    std::vector<Eigen::Vector3d> acc_buf;
    std::vector<Eigen::Vector3d> gyr_buf;
};

#endif /* integration_base_h */

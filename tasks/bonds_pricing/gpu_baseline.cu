// gpu_baseline.cu -- bonds_pricing GPU baseline
//
// Faithful port of FinanceBench Bonds/CUDA/bondsKernelsGpu.cu
// Original struct names, original GPU function names with Gpu suffix.
// One thread per bond, persistent GPU memory.

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef double dataType;

#define SIMPLE_INTEREST 0
#define COMPOUNDED_INTEREST 1
#define CONTINUOUS_INTEREST 2
#define SIMPLE_THEN_COMPOUNDED_INTEREST 3

#define ANNUAL_FREQ 1
#define SEMIANNUAL_FREQ 2

#define USE_EXACT_DAY 0
#define USE_SERIAL_NUMS 1

#define QL_EPSILON_GPU 0.000000000000000001f

#define COMPUTE_AMOUNT -1

#define MIN(a, b)  (((a) < (b)) ? (a) : (b))
#define MAX(a, b)  (((a) > (b)) ? (a) : (b))

#define ACCURACY 1.0e-8

#define MAX_LEGS 128

// ===== Structs (from bondsStructs.cuh) =====

typedef struct
{
	int month;
	int day;
	int year;
	int dateSerialNum;
} bondsDateStruct;

typedef struct
{
	bondsDateStruct startDate;
	bondsDateStruct maturityDate;
	float rate;
} bondStruct;

typedef struct
{
	dataType rate;
	dataType freq;
	int comp;
	int dayCounter;
} intRateStruct;

typedef struct
{
	dataType forward;
	dataType compounding;
	dataType frequency;
	intRateStruct intRate;
	bondsDateStruct refDate;
	bondsDateStruct calDate;
	int dayCounter;
} bondsYieldTermStruct;

typedef struct
{
	bondsDateStruct paymentDate;
	bondsDateStruct accrualStartDate;
	bondsDateStruct accrualEndDate;
	dataType amount;
} couponStruct;

typedef struct
{
	couponStruct* legs;
	intRateStruct intRate;
	int nominal;
	int dayCounter;
} cashFlowsStruct;

typedef struct
{
	dataType* dirtyPrice;
	dataType* accruedAmountCurrDate;
	dataType* cleanPrice;
	dataType* bondForwardVal;
} resultsStruct;

typedef struct
{
	bondsYieldTermStruct* discountCurve;
	bondsYieldTermStruct* repoCurve;
	bondsDateStruct* currDate;
	bondsDateStruct* maturityDate;
	dataType* bondCleanPrice;
	bondStruct* bond;
	dataType* dummyStrike;
} inArgsStruct;

typedef struct
{
	dataType npv;
	int dayCounter;
	int comp;
	dataType freq;
	bool includecurrDateFlows;
	bondsDateStruct currDate;
	bondsDateStruct npvDate;
} irrFinderStruct;

typedef struct
{
	dataType root_;
	dataType xMin_;
	dataType xMax_;
	dataType fxMin_;
	dataType fxMax_;
	int maxEvaluations_;
	int evaluationNumber_;
	dataType lowerBound_;
	dataType upperBound_;
	bool lowerBoundEnforced_;
	bool upperBoundEnforced_;
} solverStruct;


// ===== __device__ functions (from bondsKernelsGpu.cu) =====

__device__ int monthLengthKernelGpu(int month, bool leapYear)
{
	int MonthLength[12];
	MonthLength[0]=31; MonthLength[1]=28; MonthLength[2]=31;
	MonthLength[3]=30; MonthLength[4]=31; MonthLength[5]=30;
	MonthLength[6]=31; MonthLength[7]=31; MonthLength[8]=30;
	MonthLength[9]=31; MonthLength[10]=30; MonthLength[11]=31;

	int MonthLeapLength[12];
	MonthLeapLength[0]=31; MonthLeapLength[1]=29; MonthLeapLength[2]=31;
	MonthLeapLength[3]=30; MonthLeapLength[4]=31; MonthLeapLength[5]=30;
	MonthLeapLength[6]=31; MonthLeapLength[7]=31; MonthLeapLength[8]=30;
	MonthLeapLength[9]=31; MonthLeapLength[10]=30; MonthLeapLength[11]=31;

	return (leapYear? MonthLeapLength[month-1] : MonthLength[month-1]);
}

__device__ int monthOffsetKernelGpu(int m, bool leapYear)
{
	int MonthOffset[13];
	MonthOffset[0]=0; MonthOffset[1]=31; MonthOffset[2]=59;
	MonthOffset[3]=90; MonthOffset[4]=120; MonthOffset[5]=151;
	MonthOffset[6]=181; MonthOffset[7]=212; MonthOffset[8]=243;
	MonthOffset[9]=273; MonthOffset[10]=304; MonthOffset[11]=334;
	MonthOffset[12]=365;

	int MonthLeapOffset[13];
	MonthLeapOffset[0]=0; MonthLeapOffset[1]=31; MonthLeapOffset[2]=60;
	MonthLeapOffset[3]=91; MonthLeapOffset[4]=121; MonthLeapOffset[5]=152;
	MonthLeapOffset[6]=182; MonthLeapOffset[7]=213; MonthLeapOffset[8]=244;
	MonthLeapOffset[9]=274; MonthLeapOffset[10]=305; MonthLeapOffset[11]=335;
	MonthLeapOffset[12]=366;

	return (leapYear? MonthLeapOffset[m-1] : MonthOffset[m-1]);
}

__device__ int yearOffsetKernelGpu(int y)
{
	int YearOffset[121];
	YearOffset[0]=0; YearOffset[1]=366; YearOffset[2]=731;
	YearOffset[3]=1096; YearOffset[4]=1461; YearOffset[5]=1827;
	YearOffset[6]=2192; YearOffset[7]=2557; YearOffset[8]=2922;
	YearOffset[9]=3288; YearOffset[10]=3653; YearOffset[11]=4018;
	YearOffset[12]=4383; YearOffset[13]=4749; YearOffset[14]=5114;
	YearOffset[15]=5479; YearOffset[16]=5844; YearOffset[17]=6210;
	YearOffset[18]=6575; YearOffset[19]=6940; YearOffset[20]=7305;
	YearOffset[21]=7671;
	YearOffset[22]=8036; YearOffset[23]=8401; YearOffset[24]=8766;
	YearOffset[25]=9132; YearOffset[26]=9497; YearOffset[27]=9862;
	YearOffset[28]=10227; YearOffset[29]=10593; YearOffset[30]=10958;
	YearOffset[31]=11323; YearOffset[32]=11688; YearOffset[33]=12054;
	YearOffset[34]=12419; YearOffset[35]=12784; YearOffset[36]=13149;
	YearOffset[37]=13515; YearOffset[38]=13880; YearOffset[39]=14245;
	YearOffset[40]=14610; YearOffset[41]=14976; YearOffset[42]=15341;
	YearOffset[43]=15706; YearOffset[44]=16071; YearOffset[45]=16437;
	YearOffset[46]=16802; YearOffset[47]=17167; YearOffset[48]=17532;
	YearOffset[49]=17898; YearOffset[50]=18263; YearOffset[51]=18628;
	YearOffset[52]=18993; YearOffset[53]=19359; YearOffset[54]=19724;
	YearOffset[55]=20089; YearOffset[56]=20454; YearOffset[57]=20820;
	YearOffset[58]=21185; YearOffset[59]=21550; YearOffset[60]=21915;
	YearOffset[61]=22281; YearOffset[62]=22646; YearOffset[63]=23011;
	YearOffset[64]=23376; YearOffset[65]=23742; YearOffset[66]=24107;
	YearOffset[67]=24472; YearOffset[68]=24837; YearOffset[69]=25203;
	YearOffset[70]=25568; YearOffset[71]=25933; YearOffset[72]=26298;
	YearOffset[73]=26664; YearOffset[74]=27029; YearOffset[75]=27394;
	YearOffset[76]=27759; YearOffset[77]=28125; YearOffset[78]=28490;
	YearOffset[79]=28855; YearOffset[80]=29220; YearOffset[81]=29586;
	YearOffset[82]=29951; YearOffset[83]=30316; YearOffset[84]=30681;
	YearOffset[85]=31047; YearOffset[86]=31412; YearOffset[87]=31777;
	YearOffset[88]=32142; YearOffset[89]=32508; YearOffset[90]=32873;
	YearOffset[91]=33238; YearOffset[92]=33603; YearOffset[93]=33969;
	YearOffset[94]=34334; YearOffset[95]=34699; YearOffset[96]=35064;
	YearOffset[97]=35430; YearOffset[98]=35795; YearOffset[99]=36160;
	YearOffset[100]=36525; YearOffset[101]=36891; YearOffset[102]=37256;
	YearOffset[103]=37621; YearOffset[104]=37986; YearOffset[105]=38352;
	YearOffset[106]=38717; YearOffset[107]=39082; YearOffset[108]=39447;
	YearOffset[109]=39813; YearOffset[110]=40178; YearOffset[111]=40543;
	YearOffset[112]=40908; YearOffset[113]=41274; YearOffset[114]=41639;
	YearOffset[115]=42004; YearOffset[116]=42369; YearOffset[117]=42735;
	YearOffset[118]=43100; YearOffset[119]=42735; YearOffset[120]=43830;

	return YearOffset[y-1900];
}

__device__ bool isLeapKernelGpu(int y)
{
	bool YearIsLeap[121];
	YearIsLeap[0]=1; YearIsLeap[1]=0; YearIsLeap[2]=0;
	YearIsLeap[3]=0; YearIsLeap[4]=1; YearIsLeap[5]=0;
	YearIsLeap[6]=0; YearIsLeap[7]=0; YearIsLeap[8]=1;
	YearIsLeap[9]=0; YearIsLeap[10]=0; YearIsLeap[11]=0;
	YearIsLeap[12]=1; YearIsLeap[13]=0; YearIsLeap[14]=0;
	YearIsLeap[15]=0; YearIsLeap[16]=1; YearIsLeap[17]=0;
	YearIsLeap[18]=0; YearIsLeap[19]=0; YearIsLeap[20]=1;
	YearIsLeap[21]=0;
	YearIsLeap[22]=0; YearIsLeap[23]=0; YearIsLeap[24]=1;
	YearIsLeap[25]=0; YearIsLeap[26]=0; YearIsLeap[27]=0;
	YearIsLeap[28]=1; YearIsLeap[29]=0; YearIsLeap[30]=0;
	YearIsLeap[31]=0; YearIsLeap[32]=1; YearIsLeap[33]=0;
	YearIsLeap[34]=0; YearIsLeap[35]=0; YearIsLeap[36]=1;
	YearIsLeap[37]=0; YearIsLeap[38]=0; YearIsLeap[39]=0;
	YearIsLeap[40]=1; YearIsLeap[41]=0; YearIsLeap[42]=0;
	YearIsLeap[43]=0; YearIsLeap[44]=1; YearIsLeap[45]=0;
	YearIsLeap[46]=0; YearIsLeap[47]=0; YearIsLeap[48]=1;
	YearIsLeap[49]=0; YearIsLeap[50]=0; YearIsLeap[51]=0;
	YearIsLeap[52]=1; YearIsLeap[53]=0; YearIsLeap[54]=0;
	YearIsLeap[55]=0; YearIsLeap[56]=1; YearIsLeap[57]=0;
	YearIsLeap[58]=0; YearIsLeap[59]=0; YearIsLeap[60]=1;
	YearIsLeap[61]=0; YearIsLeap[62]=0; YearIsLeap[63]=0;
	YearIsLeap[64]=1; YearIsLeap[65]=0; YearIsLeap[66]=0;
	YearIsLeap[67]=0; YearIsLeap[68]=1; YearIsLeap[69]=0;
	YearIsLeap[70]=0; YearIsLeap[71]=0; YearIsLeap[72]=1;
	YearIsLeap[73]=0; YearIsLeap[74]=0; YearIsLeap[75]=0;
	YearIsLeap[76]=1; YearIsLeap[77]=0; YearIsLeap[78]=0;
	YearIsLeap[79]=0; YearIsLeap[80]=1; YearIsLeap[81]=0;
	YearIsLeap[82]=0; YearIsLeap[83]=0; YearIsLeap[84]=1;
	YearIsLeap[85]=0; YearIsLeap[86]=0; YearIsLeap[87]=0;
	YearIsLeap[88]=1; YearIsLeap[89]=0; YearIsLeap[90]=0;
	YearIsLeap[91]=0; YearIsLeap[92]=1; YearIsLeap[93]=0;
	YearIsLeap[94]=0; YearIsLeap[95]=0; YearIsLeap[96]=1;
	YearIsLeap[97]=0; YearIsLeap[98]=0; YearIsLeap[99]=0;
	YearIsLeap[100]=1; YearIsLeap[101]=0; YearIsLeap[102]=0;
	YearIsLeap[103]=0; YearIsLeap[104]=1; YearIsLeap[105]=0;
	YearIsLeap[106]=0; YearIsLeap[107]=0; YearIsLeap[108]=1;
	YearIsLeap[109]=0; YearIsLeap[110]=0; YearIsLeap[111]=0;
	YearIsLeap[112]=1; YearIsLeap[113]=0; YearIsLeap[114]=0;
	YearIsLeap[115]=0; YearIsLeap[116]=1; YearIsLeap[117]=0;
	YearIsLeap[118]=0; YearIsLeap[119]=0; YearIsLeap[120]=1;

	return YearIsLeap[y-1900];
}

__device__ bondsDateStruct intializeDateKernelGpu(int d, int m, int y)
{
	bondsDateStruct currDate;
	currDate.day = d;
	currDate.month = m;
	currDate.year = y;
	bool leap = isLeapKernelGpu(y);
	int offset = monthOffsetKernelGpu(m, leap);
	currDate.dateSerialNum = d + offset + yearOffsetKernelGpu(y);
	return currDate;
}

__device__ dataType yearFractionGpu(bondsDateStruct d1, bondsDateStruct d2, int dayCounter)
{
	return dayCountGpu(d1, d2, dayCounter) / 360.0;
}

__device__ int dayCountGpu(bondsDateStruct d1, bondsDateStruct d2, int dayCounter)
{
	if (dayCounter == USE_EXACT_DAY)
	{
		int dd1 = d1.day, dd2 = d2.day;
		int mm1 = d1.month, mm2 = d2.month;
		int yy1 = d1.year, yy2 = d2.year;
		if (dd2 == 31 && dd1 < 30) { dd2 = 1; mm2++; }
		return 360*(yy2-yy1) + 30*(mm2-mm1-1) + MAX(0, 30-dd1) + MIN(30, dd2);
	}
	else
	{
		return (d2.dateSerialNum - d1.dateSerialNum);
	}
}

__device__ dataType couponNotionalGpu() { return 100.0; }
__device__ dataType bondNotionalGpu() { return 100.0; }
__device__ dataType fixedRateCouponNominalGpu() { return 100.0; }

__device__ bool eventHasOccurredGpu(bondsDateStruct currDate, bondsDateStruct eventDate)
{
	return eventDate.dateSerialNum > currDate.dateSerialNum;
}

__device__ bool cashFlowHasOccurredGpu(bondsDateStruct refDate, bondsDateStruct eventDate)
{
	return eventHasOccurredGpu(refDate, eventDate);
}

__device__ bondsDateStruct advanceDateGpu(bondsDateStruct date, int numMonthsAdvance)
{
	int d = date.day;
	int m = date.month + numMonthsAdvance;
	int y = date.year;
	while (m > 12) { m -= 12; y += 1; }
	while (m < 1)  { m += 12; y -= 1; }
	int length = monthLengthKernelGpu(m, isLeapKernelGpu(y));
	if (d > length) d = length;
	bondsDateStruct newDate = intializeDateKernelGpu(d, m, y);
	return newDate;
}

__device__ int getNumCashFlowsGpu(inArgsStruct inArgs, int bondNum)
{
	int numCashFlows = 0;
	bondsDateStruct currCashflowDate = inArgs.bond[bondNum].maturityDate;
	while (currCashflowDate.dateSerialNum > inArgs.bond[bondNum].startDate.dateSerialNum)
	{
		numCashFlows++;
		currCashflowDate = advanceDateGpu(currCashflowDate, -6);
	}
	return numCashFlows+1;
}

// Forward declarations for mutual recursion
__device__ dataType interestRateCompoundFactorGpuTwoArgs(intRateStruct intRate, dataType t);
__device__ dataType interestRateCompoundFactorGpu(intRateStruct intRate, bondsDateStruct d1, bondsDateStruct d2, int dayCounter);
__device__ dataType fixedRateCouponAmountGpu(cashFlowsStruct cashFlows, int numLeg);
__device__ dataType bondsYieldTermStructureDiscountGpu(bondsYieldTermStruct ytStruct, bondsDateStruct t);
__device__ dataType fOpGpu(irrFinderStruct f, dataType y, cashFlowsStruct cashFlows, int numLegs);
__device__ dataType fDerivativeGpu(irrFinderStruct f, dataType y, cashFlowsStruct cashFlows, int numLegs);
__device__ bool closeGpu(dataType x, dataType y);
__device__ bool closeGpuThreeArgs(dataType x, dataType y, int n);
__device__ dataType solveImplGpu(solverStruct solver, irrFinderStruct f, dataType xAccuracy, cashFlowsStruct cashFlows, int numLegs);


__device__ dataType getDirtyPriceGpu(inArgsStruct inArgs, int bondNum, cashFlowsStruct cashFlows, int numLegs)
{
	dataType currentNotional = bondNotionalGpu();
	return discountingBondEngineCalculateSettlementValueGpu(inArgs, bondNum, cashFlows, numLegs) * 100.0 / currentNotional;
}

__device__ dataType getAccruedAmountGpu(inArgsStruct inArgs, bondsDateStruct date, int bondNum, cashFlowsStruct cashFlows, int numLegs)
{
	return bondAccruedAmountGpu(inArgs, date, bondNum, cashFlows, numLegs);
}

__device__ dataType discountingBondEngineCalculateSettlementValueGpu(inArgsStruct inArgs, int bondNum, cashFlowsStruct cashFlows, int numLegs)
{
	bondsDateStruct currDate = inArgs.currDate[bondNum];
	if (currDate.dateSerialNum < inArgs.bond[bondNum].startDate.dateSerialNum)
	{
		currDate = inArgs.bond[bondNum].startDate;
	}
	return cashFlowsNpvGpu(cashFlows, inArgs.discountCurve[bondNum], false, currDate, currDate, numLegs);
}

__device__ dataType bondAccruedAmountGpu(inArgsStruct inArgs, bondsDateStruct date, int bondNum, cashFlowsStruct cashFlows, int numLegs)
{
	dataType currentNotional = bondNotionalGpu();
	if (currentNotional == 0.0)
		return 0.0;
	return bondFunctionsAccruedAmountGpu(inArgs, date, bondNum, cashFlows, numLegs);
}

__device__ dataType bondFunctionsAccruedAmountGpu(inArgsStruct inArgs, bondsDateStruct date, int bondNum, cashFlowsStruct cashFlows, int numLegs)
{
	return cashFlowsAccruedAmountGpu(cashFlows, false, date, numLegs, inArgs, bondNum) *
	    100.0 / bondNotionalGpu();
}

__device__ int cashFlowsNextCashFlowNumGpu(cashFlowsStruct cashFlows, bondsDateStruct currDate, int numLegs)
{
	int i;
	for (i = 0; i < numLegs; ++i)
	{
		if ( ! (cashFlowHasOccurredGpu(cashFlows.legs[i].paymentDate, currDate) ))
			return i;
	}
	return (numLegs-1);
}

__device__ couponStruct cashFlowsNextCashFlowGpu(cashFlowsStruct cashFlows, bondsDateStruct currDate, int numLegs)
{
	int i;
	for (i = 0; i < numLegs; ++i)
	{
		if ( ! (cashFlowHasOccurredGpu(cashFlows.legs[i].paymentDate, currDate) ))
			return cashFlows.legs[i];
	}
	return cashFlows.legs[numLegs-1];
}

__device__ dataType cashFlowsAccruedAmountGpu(cashFlowsStruct cashFlows,
                                  bool includecurrDateFlows,
                                  bondsDateStruct currDate,
                                  int numLegs, inArgsStruct inArgs, int bondNum)
{
	int legComputeNum = cashFlowsNextCashFlowNumGpu(cashFlows, currDate, numLegs);
	dataType result = 0.0;
	int i;
	for (i = legComputeNum; i < numLegs; ++i)
	{
		result += fixedRateCouponAccruedAmountGpu(cashFlows, i, currDate, inArgs, bondNum);
	}
	return result;
}

__device__ dataType fixedRateCouponAccruedAmountGpu(cashFlowsStruct cashFlows, int numLeg, bondsDateStruct d, inArgsStruct inArgs, int bondNum)
{
	if (d.dateSerialNum <= cashFlows.legs[numLeg].accrualStartDate.dateSerialNum || d.dateSerialNum > inArgs.maturityDate[bondNum].dateSerialNum)
	{
		return 0.0;
	}
	else
	{
		bondsDateStruct endDate = cashFlows.legs[numLeg].accrualEndDate;
		if (d.dateSerialNum < cashFlows.legs[numLeg].accrualEndDate.dateSerialNum)
		{
			endDate = d;
		}
		return fixedRateCouponNominalGpu()*(interestRateCompoundFactorGpu(cashFlows.intRate, cashFlows.legs[numLeg].accrualStartDate, endDate, cashFlows.dayCounter) - 1.0);
	}
}

__device__ dataType cashFlowsNpvGpu(cashFlowsStruct cashFlows,
                        bondsYieldTermStruct discountCurve,
                        bool includecurrDateFlows,
                        bondsDateStruct currDate,
                        bondsDateStruct npvDate,
                        int numLegs)
{
	npvDate = currDate;
	dataType totalNPV = 0.0;
	int i;
	for (i=0; i<numLegs; ++i)
	{
		if (!(cashFlowHasOccurredGpu(cashFlows.legs[i].paymentDate, currDate)))
			totalNPV += fixedRateCouponAmountGpu(cashFlows, i) *
			            bondsYieldTermStructureDiscountGpu(discountCurve, cashFlows.legs[i].paymentDate);
	}
	return totalNPV/bondsYieldTermStructureDiscountGpu(discountCurve, npvDate);
}

__device__ dataType bondsYieldTermStructureDiscountGpu(bondsYieldTermStruct ytStruct, bondsDateStruct t)
{
	ytStruct.intRate.rate = ytStruct.forward;
	ytStruct.intRate.freq = ytStruct.frequency;
	ytStruct.intRate.comp = ytStruct.compounding;
	return flatForwardDiscountImplGpu(ytStruct.intRate, yearFractionGpu(ytStruct.refDate, t, ytStruct.dayCounter));
}

__device__ dataType flatForwardDiscountImplGpu(intRateStruct intRate, dataType t)
{
	return interestRateDiscountFactorGpu(intRate, t);
}

__device__ dataType interestRateDiscountFactorGpu(intRateStruct intRate, dataType t)
{
	return 1.0/interestRateCompoundFactorGpuTwoArgs(intRate, t);
}

__device__ dataType interestRateCompoundFactorGpuTwoArgs(intRateStruct intRate, dataType t)
{
	{
		if (intRate.comp == SIMPLE_INTEREST)
			return 1.0 + intRate.rate*t;
		else if (intRate.comp == COMPOUNDED_INTEREST)
			return pow(1.0f+intRate.rate/intRate.freq, intRate.freq*t);
		else if (intRate.comp == CONTINUOUS_INTEREST)
			return exp(intRate.rate*t);
	}
	return 0.0f;
}

__device__ dataType fixedRateCouponAmountGpu(cashFlowsStruct cashFlows, int numLeg)
{
	if (cashFlows.legs[numLeg].amount == COMPUTE_AMOUNT)
	{
		return fixedRateCouponNominalGpu()*(interestRateCompoundFactorGpu(cashFlows.intRate, cashFlows.legs[numLeg].accrualStartDate,
                                              cashFlows.legs[numLeg].accrualEndDate, cashFlows.dayCounter) - 1.0);
	}
	else
	{
		return cashFlows.legs[numLeg].amount;
	}
}

__device__ dataType interestRateCompoundFactorGpu(intRateStruct intRate, bondsDateStruct d1,
                                           bondsDateStruct d2, int dayCounter)
{
	dataType t = yearFractionGpu(d1, d2, dayCounter);
	return interestRateCompoundFactorGpuTwoArgs(intRate, t);
}

__device__ dataType interestRateImpliedRateGpu(dataType compound,
                              int comp,
                              dataType freq,
                              dataType t)
{
	dataType r = 0.0f;
	if (compound==1.0)
	{
		r = 0.0;
	}
	else
	{
		switch (comp)
		{
			case SIMPLE_INTEREST:
				r = (compound - 1.0)/t;
				break;
			case COMPOUNDED_INTEREST:
				r = (pow((dataType)compound, 1.0f/((freq)*t))-1.0f)*(freq);
				break;
		}
	}
	return r;
}

__device__ dataType getMarketRepoRateGpu(bondsDateStruct d,
                                   int comp,
                                   dataType freq,
                                   bondsDateStruct referenceDate,
                                   inArgsStruct inArgs, int bondNum)
{
	dataType compound = 1.0/bondsYieldTermStructureDiscountGpu(inArgs.repoCurve[bondNum], d);
	return interestRateImpliedRateGpu(compound,
	                                   comp, freq,
	                                   yearFractionGpu(referenceDate, d, inArgs.repoCurve[bondNum].dayCounter));
}

__device__ dataType getBondYieldGpu(dataType cleanPrice,
                     int dc, int comp, dataType freq,
                     bondsDateStruct settlement,
                     dataType accuracy, int maxEvaluations,
                     inArgsStruct currInArgs, int bondNum, cashFlowsStruct cashFlows, int numLegs)
{
	dataType currentNotional = bondNotionalGpu();
	if (currentNotional == 0.0) return 0.0;
	if (currInArgs.bond[bondNum].startDate.dateSerialNum > settlement.dateSerialNum)
	{
		settlement = currInArgs.bond[bondNum].startDate;
	}
	return getBondFunctionsYieldGpu(cleanPrice, dc, comp, freq,
	                               settlement, accuracy, maxEvaluations,
	                               currInArgs, bondNum, cashFlows, numLegs);
}

__device__ dataType getBondFunctionsYieldGpu(dataType cleanPrice,
                     int dc, int comp, dataType freq,
                     bondsDateStruct settlement,
                     dataType accuracy, int maxEvaluations,
                     inArgsStruct currInArgs, int bondNum, cashFlowsStruct cashFlows, int numLegs)
{
	dataType dirtyPrice = cleanPrice + bondFunctionsAccruedAmountGpu(currInArgs, settlement, bondNum, cashFlows, numLegs);
	dirtyPrice /= 100.0 / bondNotionalGpu();
	return getCashFlowsYieldGpu(cashFlows, dirtyPrice,
	                            dc, comp, freq,
	                            false, settlement, settlement, numLegs,
	                            accuracy, maxEvaluations, 0.05f);
}

__device__ dataType getCashFlowsYieldGpu(cashFlowsStruct leg,
                          dataType npv,
                          int dayCounter,
                          int compounding,
                          dataType frequency,
                          bool includecurrDateFlows,
                          bondsDateStruct currDate,
                          bondsDateStruct npvDate,
                          int numLegs,
                          dataType accuracy,
                          int maxIterations,
                          dataType guess)
{
	solverStruct solver;
	solver.maxEvaluations_ = maxIterations;
	irrFinderStruct objFunction;

	objFunction.npv = npv;
	objFunction.dayCounter = dayCounter;
	objFunction.comp = compounding;
	objFunction.freq = frequency;
	objFunction.includecurrDateFlows = includecurrDateFlows;
	objFunction.currDate = currDate;
	objFunction.npvDate = npvDate;

	return solverSolveGpu(solver, objFunction, accuracy, guess, guess/10.0, leg, numLegs);
}

__device__ dataType solverSolveGpu(solverStruct solver,
                        irrFinderStruct f,
                        dataType accuracy,
                        dataType guess,
                        dataType step,
                        cashFlowsStruct cashFlows,
                        int numLegs)
{
	accuracy = MAX(accuracy, QL_EPSILON_GPU);
	dataType growthFactor = 1.6;
	int flipflop = -1;

	solver.root_ = guess;
	solver.fxMax_ = fOpGpu(f, solver.root_, cashFlows, numLegs);

	if (closeGpu(solver.fxMax_,0.0))
	{
		return solver.root_;
	}
	else if (closeGpu(solver.fxMax_, 0.0))
	{
		solver.xMin_ = (solver.root_ - step);
		solver.fxMin_ = fOpGpu(f, solver.xMin_, cashFlows, numLegs);
		solver.xMax_ = solver.root_;
	}
	else
	{
		solver.xMin_ = solver.root_;
		solver.fxMin_ = solver.fxMax_;
		solver.xMax_ = (solver.root_+step);
		solver.fxMax_ = fOpGpu(f, solver.xMax_, cashFlows, numLegs);
	}

	solver.evaluationNumber_ = 2;
	while (solver.evaluationNumber_ <= solver.maxEvaluations_)
	{
		if (solver.fxMin_*solver.fxMax_ <= 0.0)
		{
			if (closeGpu(solver.fxMin_, 0.0))
				return solver.xMin_;
			if (closeGpu(solver.fxMax_, 0.0))
				return solver.xMax_;
			solver.root_ = (solver.xMax_+solver.xMin_)/2.0;
			return solveImplGpu(solver, f, accuracy, cashFlows, numLegs);
		}
		if (fabs(solver.fxMin_) < fabs(solver.fxMax_))
		{
			solver.xMin_ = (solver.xMin_+growthFactor*(solver.xMin_ - solver.xMax_));
			solver.fxMin_= fOpGpu(f, solver.xMin_, cashFlows, numLegs);
		}
		else if (fabs(solver.fxMin_) > fabs(solver.fxMax_))
		{
			solver.xMax_ = (solver.xMax_+growthFactor*(solver.xMax_ - solver.xMin_));
			solver.fxMax_= fOpGpu(f, solver.xMax_, cashFlows, numLegs);
		}
		else if (flipflop == -1)
		{
			solver.xMin_ = (solver.xMin_+growthFactor*(solver.xMin_ - solver.xMax_));
			solver.fxMin_= fOpGpu(f, solver.xMin_, cashFlows, numLegs);
			solver.evaluationNumber_++;
			flipflop = 1;
		}
		else if (flipflop == 1)
		{
			solver.xMax_ = (solver.xMax_+growthFactor*(solver.xMax_ - solver.xMin_));
			solver.fxMax_= fOpGpu(f, solver.xMax_, cashFlows, numLegs);
			flipflop = -1;
		}
		solver.evaluationNumber_++;
	}
	return 0.0f;
}

__device__ dataType cashFlowsNpvYieldGpu(cashFlowsStruct cashFlows,
                        intRateStruct y,
                        bool includecurrDateFlows,
                        bondsDateStruct currDate,
                        bondsDateStruct npvDate,
                        int numLegs)
{
	dataType npv = 0.0;
	dataType discount = 1.0;
	bondsDateStruct lastDate;
	bool first = true;
	int i;
	for (i=0; i<numLegs; ++i)
	{
		if (cashFlowHasOccurredGpu(cashFlows.legs[i].paymentDate, currDate))
			continue;

		bondsDateStruct couponDate = cashFlows.legs[i].paymentDate;
		dataType amount = fixedRateCouponAmountGpu(cashFlows, i);
		if (first)
		{
			first = false;
			if (i > 0) {
				lastDate = advanceDateGpu(cashFlows.legs[i].paymentDate, -1*6);
			} else {
				lastDate = cashFlows.legs[i].accrualStartDate;
			}
			discount *= interestRateDiscountFactorGpu(y, yearFractionGpu(npvDate, couponDate, y.dayCounter));
		}
		else
		{
			discount *= interestRateDiscountFactorGpu(y, yearFractionGpu(lastDate, couponDate, y.dayCounter));
		}
		lastDate = couponDate;
		npv += amount * discount;
	}
	return npv;
}

__device__ dataType fOpGpu(irrFinderStruct f, dataType y, cashFlowsStruct cashFlows, int numLegs)
{
	intRateStruct yield;
	yield.rate = y;
	yield.comp = f.comp;
	yield.freq = f.freq;
	yield.dayCounter = f.dayCounter;
	dataType NPV = cashFlowsNpvYieldGpu(cashFlows, yield, false, f.currDate, f.npvDate, numLegs);
	return (f.npv - NPV);
}

__device__ dataType fDerivativeGpu(irrFinderStruct f, dataType y, cashFlowsStruct cashFlows, int numLegs)
{
	intRateStruct yield;
	yield.rate = y;
	yield.dayCounter = f.dayCounter;
	yield.comp = f.comp;
	yield.freq = f.freq;
	return modifiedDurationGpu(cashFlows, yield, f.includecurrDateFlows, f.currDate, f.npvDate, numLegs);
}

__device__ bool closeGpu(dataType x, dataType y)
{
	return closeGpuThreeArgs(x, y, 42);
}

__device__ bool closeGpuThreeArgs(dataType x, dataType y, int n)
{
	dataType diff = fabs(x-y);
	dataType tolerance = n*QL_EPSILON_GPU;
	return diff <= tolerance*fabs(x) &&
	       diff <= tolerance*fabs(y);
}

__device__ dataType enforceBoundsGpu(dataType x)
{
	return x;
}

__device__ dataType solveImplGpu(solverStruct solver, irrFinderStruct f,
                        dataType xAccuracy, cashFlowsStruct cashFlows, int numLegs)
{
	dataType froot, dfroot, dx, dxold;
	dataType xh, xl;

	if (solver.fxMin_ < 0.0)
	{
		xl = solver.xMin_;
		xh = solver.xMax_;
	}
	else
	{
		xh = solver.xMin_;
		xl = solver.xMax_;
	}

	dxold = solver.xMax_ - solver.xMin_;
	dx = dxold;

	froot = fOpGpu(f, solver.root_, cashFlows, numLegs);
	dfroot = fDerivativeGpu(f, solver.root_, cashFlows, numLegs);
	++solver.evaluationNumber_;

	while (solver.evaluationNumber_<=solver.maxEvaluations_)
	{
		if ((((solver.root_-xh)*dfroot-froot)*
			((solver.root_-xl)*dfroot-froot) > 0.0)
			|| (fabs(2.0*froot) > fabs(dxold*dfroot)))
		{
			dxold = dx;
			dx = (xh-xl)/2.0;
			solver.root_=xl+dx;
		}
		else
		{
			dxold = dx;
			dx = froot/dfroot;
			solver.root_ -= dx;
		}

		if (fabs(dx) < xAccuracy)
			return solver.root_;
		froot = fOpGpu(f, solver.root_, cashFlows, numLegs);
		dfroot = fDerivativeGpu(f, solver.root_, cashFlows, numLegs);
		++solver.evaluationNumber_;
		if (froot < 0.0)
			xl=solver.root_;
		else
			xh=solver.root_;
	}
	return solver.root_;
}

__device__ dataType modifiedDurationGpu(cashFlowsStruct cashFlows,
                                intRateStruct y,
                                bool includecurrDateFlows,
                                bondsDateStruct currDate,
                                bondsDateStruct npvDate,
                                int numLegs)
{
	dataType P = 0.0;
	dataType dPdy = 0.0;
	dataType r = y.rate;
	dataType N = y.freq;
	int dc = y.dayCounter;

	int i;
	for (i=0; i<numLegs; ++i)
	{
		if (!cashFlowHasOccurredGpu(cashFlows.legs[i].paymentDate, currDate))
		{
			dataType t = yearFractionGpu(npvDate, cashFlows.legs[i].paymentDate, dc);
			dataType c = fixedRateCouponAmountGpu(cashFlows, i);
			dataType B = interestRateDiscountFactorGpu(y, t);

			P += c * B;
			{
				if (y.comp == SIMPLE_INTEREST)
					dPdy -= c * B*B * t;
				if (y.comp == COMPOUNDED_INTEREST)
					dPdy -= c * t * B/(1+r/N);
				if (y.comp == CONTINUOUS_INTEREST)
					dPdy -= c * B * t;
				if (y.comp == SIMPLE_THEN_COMPOUNDED_INTEREST)
				{
					if (t<=1.0/N)
						dPdy -= c * B*B * t;
					else
						dPdy -= c * t * B/(1+r/N);
				}
			}
		}
	}

	if (P == 0.0)
		return 0.0;
	return (-1*dPdy)/P;
}


// ===== __global__ kernel (from bondsKernelsGpu.cu: getBondsResultsGpu) =====

__global__ __launch_bounds__(256)
void getBondsResultsGpuKernel(inArgsStruct inArgs, resultsStruct results, int n)
{
	int bondNum = blockIdx.x * blockDim.x + threadIdx.x;
	if (bondNum < n)
	{
		int numLegs;
		int numCashFlows = 0;
		bondsDateStruct currCashflowDate = inArgs.bond[bondNum].maturityDate;

		while (currCashflowDate.dateSerialNum > inArgs.bond[bondNum].startDate.dateSerialNum)
		{
			numCashFlows++;
			currCashflowDate = advanceDateGpu(currCashflowDate, -6);
		}
		numLegs = numCashFlows+1;

		cashFlowsStruct cashFlows;
		couponStruct cashLegs[MAX_LEGS];
		cashFlows.legs = cashLegs;

		cashFlows.intRate.dayCounter = USE_EXACT_DAY;
		cashFlows.intRate.rate  = inArgs.bond[bondNum].rate;
		cashFlows.intRate.freq  = ANNUAL_FREQ;
		cashFlows.intRate.comp  = SIMPLE_INTEREST;
		cashFlows.dayCounter  = USE_EXACT_DAY;
		cashFlows.nominal  = 100.0;

		bondsDateStruct currStartDate = advanceDateGpu(inArgs.bond[bondNum].maturityDate, (numLegs - 1)*-6);
		bondsDateStruct currEndDate = advanceDateGpu(currStartDate, 6);

		int cashFlowNum;
		for (cashFlowNum = 0; cashFlowNum < numLegs-1; cashFlowNum++)
		{
			cashFlows.legs[cashFlowNum].paymentDate = currEndDate;
			cashFlows.legs[cashFlowNum].accrualStartDate  = currStartDate;
			cashFlows.legs[cashFlowNum].accrualEndDate  = currEndDate;
			cashFlows.legs[cashFlowNum].amount = COMPUTE_AMOUNT;

			currStartDate = currEndDate;
			currEndDate = advanceDateGpu(currEndDate, 6);
		}

		cashFlows.legs[numLegs-1].paymentDate  = inArgs.bond[bondNum].maturityDate;
		cashFlows.legs[numLegs-1].accrualStartDate = inArgs.currDate[bondNum];
		cashFlows.legs[numLegs-1].accrualEndDate  = inArgs.currDate[bondNum];
		cashFlows.legs[numLegs-1].amount = 100.0;

		results.bondForwardVal[bondNum] = getBondYieldGpu(inArgs.bondCleanPrice[bondNum],
		                     USE_EXACT_DAY,
		                     COMPOUNDED_INTEREST,
		                     2.0,
		                     inArgs.currDate[bondNum],
		                     ACCURACY,
		                     100,
		                     inArgs, bondNum, cashFlows, numLegs);
		inArgs.discountCurve[bondNum].forward = results.bondForwardVal[bondNum];
		results.dirtyPrice[bondNum] = getDirtyPriceGpu(inArgs, bondNum, cashFlows, numLegs);
		results.accruedAmountCurrDate[bondNum] = getAccruedAmountGpu(inArgs, inArgs.currDate[bondNum], bondNum, cashFlows, numLegs);
		results.cleanPrice[bondNum] = results.dirtyPrice[bondNum] - results.accruedAmountCurrDate[bondNum];
	}
}


// ===== Host-side persistent GPU memory =====

// Host-side copies of inArgs pointers for CPU date initialization
static bondsDateStruct intializeDateKernelCpu(int d, int m, int y);

static int g_N = 0;
static const int* g_issue_year = NULL;
static const int* g_issue_month = NULL;
static const int* g_issue_day = NULL;
static const int* g_maturity_year = NULL;
static const int* g_maturity_month = NULL;
static const int* g_maturity_day = NULL;
static const float* g_rates = NULL;
static float g_coupon_freq = 2.0f;

// Device pointers (persistent across compute calls)
static bondsYieldTermStruct* d_discountCurve = NULL;
static bondsYieldTermStruct* d_repoCurve = NULL;
static bondsDateStruct* d_currDate = NULL;
static bondsDateStruct* d_maturityDate = NULL;
static dataType* d_bondCleanPrice = NULL;
static bondStruct* d_bond = NULL;
static dataType* d_dummyStrike = NULL;

static dataType* d_dirtyPrice = NULL;
static dataType* d_accruedAmountCurrDate = NULL;
static dataType* d_cleanPrice = NULL;
static dataType* d_bondForwardVal = NULL;

// CPU date init (same as original bondsEngine.c / cpu_reference.c)

static int monthOffsetCpu(int m, bool leapYear)
{
	int MonthOffset[] = {0,31,59,90,120,151,181,212,243,273,304,334,365};
	int MonthLeapOffset[] = {0,31,60,91,121,152,182,213,244,274,305,335,366};
	return (leapYear? MonthLeapOffset[m-1] : MonthOffset[m-1]);
}

static int yearOffsetCpu(int y)
{
	int YearOffset[121];
	YearOffset[0]=0; YearOffset[1]=366; YearOffset[2]=731;
	YearOffset[3]=1096; YearOffset[4]=1461; YearOffset[5]=1827;
	YearOffset[6]=2192; YearOffset[7]=2557; YearOffset[8]=2922;
	YearOffset[9]=3288; YearOffset[10]=3653; YearOffset[11]=4018;
	YearOffset[12]=4383; YearOffset[13]=4749; YearOffset[14]=5114;
	YearOffset[15]=5479; YearOffset[16]=5844; YearOffset[17]=6210;
	YearOffset[18]=6575; YearOffset[19]=6940; YearOffset[20]=7305;
	YearOffset[21]=7671;
	YearOffset[22]=8036; YearOffset[23]=8401; YearOffset[24]=8766;
	YearOffset[25]=9132; YearOffset[26]=9497; YearOffset[27]=9862;
	YearOffset[28]=10227; YearOffset[29]=10593; YearOffset[30]=10958;
	YearOffset[31]=11323; YearOffset[32]=11688; YearOffset[33]=12054;
	YearOffset[34]=12419; YearOffset[35]=12784; YearOffset[36]=13149;
	YearOffset[37]=13515; YearOffset[38]=13880; YearOffset[39]=14245;
	YearOffset[40]=14610; YearOffset[41]=14976; YearOffset[42]=15341;
	YearOffset[43]=15706; YearOffset[44]=16071; YearOffset[45]=16437;
	YearOffset[46]=16802; YearOffset[47]=17167; YearOffset[48]=17532;
	YearOffset[49]=17898; YearOffset[50]=18263; YearOffset[51]=18628;
	YearOffset[52]=18993; YearOffset[53]=19359; YearOffset[54]=19724;
	YearOffset[55]=20089; YearOffset[56]=20454; YearOffset[57]=20820;
	YearOffset[58]=21185; YearOffset[59]=21550; YearOffset[60]=21915;
	YearOffset[61]=22281; YearOffset[62]=22646; YearOffset[63]=23011;
	YearOffset[64]=23376; YearOffset[65]=23742; YearOffset[66]=24107;
	YearOffset[67]=24472; YearOffset[68]=24837; YearOffset[69]=25203;
	YearOffset[70]=25568; YearOffset[71]=25933; YearOffset[72]=26298;
	YearOffset[73]=26664; YearOffset[74]=27029; YearOffset[75]=27394;
	YearOffset[76]=27759; YearOffset[77]=28125; YearOffset[78]=28490;
	YearOffset[79]=28855; YearOffset[80]=29220; YearOffset[81]=29586;
	YearOffset[82]=29951; YearOffset[83]=30316; YearOffset[84]=30681;
	YearOffset[85]=31047; YearOffset[86]=31412; YearOffset[87]=31777;
	YearOffset[88]=32142; YearOffset[89]=32508; YearOffset[90]=32873;
	YearOffset[91]=33238; YearOffset[92]=33603; YearOffset[93]=33969;
	YearOffset[94]=34334; YearOffset[95]=34699; YearOffset[96]=35064;
	YearOffset[97]=35430; YearOffset[98]=35795; YearOffset[99]=36160;
	YearOffset[100]=36525; YearOffset[101]=36891; YearOffset[102]=37256;
	YearOffset[103]=37621; YearOffset[104]=37986; YearOffset[105]=38352;
	YearOffset[106]=38717; YearOffset[107]=39082; YearOffset[108]=39447;
	YearOffset[109]=39813; YearOffset[110]=40178; YearOffset[111]=40543;
	YearOffset[112]=40908; YearOffset[113]=41274; YearOffset[114]=41639;
	YearOffset[115]=42004; YearOffset[116]=42369; YearOffset[117]=42735;
	YearOffset[118]=43100; YearOffset[119]=42735; YearOffset[120]=43830;
	return YearOffset[y-1900];
}

static bool isLeapCpu(int y)
{
	bool YearIsLeap[121];
	YearIsLeap[0]=1; YearIsLeap[1]=0; YearIsLeap[2]=0;
	YearIsLeap[3]=0; YearIsLeap[4]=1; YearIsLeap[5]=0;
	YearIsLeap[6]=0; YearIsLeap[7]=0; YearIsLeap[8]=1;
	YearIsLeap[9]=0; YearIsLeap[10]=0; YearIsLeap[11]=0;
	YearIsLeap[12]=1; YearIsLeap[13]=0; YearIsLeap[14]=0;
	YearIsLeap[15]=0; YearIsLeap[16]=1; YearIsLeap[17]=0;
	YearIsLeap[18]=0; YearIsLeap[19]=0; YearIsLeap[20]=1;
	YearIsLeap[21]=0;
	YearIsLeap[22]=0; YearIsLeap[23]=0; YearIsLeap[24]=1;
	YearIsLeap[25]=0; YearIsLeap[26]=0; YearIsLeap[27]=0;
	YearIsLeap[28]=1; YearIsLeap[29]=0; YearIsLeap[30]=0;
	YearIsLeap[31]=0; YearIsLeap[32]=1; YearIsLeap[33]=0;
	YearIsLeap[34]=0; YearIsLeap[35]=0; YearIsLeap[36]=1;
	YearIsLeap[37]=0; YearIsLeap[38]=0; YearIsLeap[39]=0;
	YearIsLeap[40]=1; YearIsLeap[41]=0; YearIsLeap[42]=0;
	YearIsLeap[43]=0; YearIsLeap[44]=1; YearIsLeap[45]=0;
	YearIsLeap[46]=0; YearIsLeap[47]=0; YearIsLeap[48]=1;
	YearIsLeap[49]=0; YearIsLeap[50]=0; YearIsLeap[51]=0;
	YearIsLeap[52]=1; YearIsLeap[53]=0; YearIsLeap[54]=0;
	YearIsLeap[55]=0; YearIsLeap[56]=1; YearIsLeap[57]=0;
	YearIsLeap[58]=0; YearIsLeap[59]=0; YearIsLeap[60]=1;
	YearIsLeap[61]=0; YearIsLeap[62]=0; YearIsLeap[63]=0;
	YearIsLeap[64]=1; YearIsLeap[65]=0; YearIsLeap[66]=0;
	YearIsLeap[67]=0; YearIsLeap[68]=1; YearIsLeap[69]=0;
	YearIsLeap[70]=0; YearIsLeap[71]=0; YearIsLeap[72]=1;
	YearIsLeap[73]=0; YearIsLeap[74]=0; YearIsLeap[75]=0;
	YearIsLeap[76]=1; YearIsLeap[77]=0; YearIsLeap[78]=0;
	YearIsLeap[79]=0; YearIsLeap[80]=1; YearIsLeap[81]=0;
	YearIsLeap[82]=0; YearIsLeap[83]=0; YearIsLeap[84]=1;
	YearIsLeap[85]=0; YearIsLeap[86]=0; YearIsLeap[87]=0;
	YearIsLeap[88]=1; YearIsLeap[89]=0; YearIsLeap[90]=0;
	YearIsLeap[91]=0; YearIsLeap[92]=1; YearIsLeap[93]=0;
	YearIsLeap[94]=0; YearIsLeap[95]=0; YearIsLeap[96]=1;
	YearIsLeap[97]=0; YearIsLeap[98]=0; YearIsLeap[99]=0;
	YearIsLeap[100]=1; YearIsLeap[101]=0; YearIsLeap[102]=0;
	YearIsLeap[103]=0; YearIsLeap[104]=1; YearIsLeap[105]=0;
	YearIsLeap[106]=0; YearIsLeap[107]=0; YearIsLeap[108]=1;
	YearIsLeap[109]=0; YearIsLeap[110]=0; YearIsLeap[111]=0;
	YearIsLeap[112]=1; YearIsLeap[113]=0; YearIsLeap[114]=0;
	YearIsLeap[115]=0; YearIsLeap[116]=1; YearIsLeap[117]=0;
	YearIsLeap[118]=0; YearIsLeap[119]=0; YearIsLeap[120]=1;
	return YearIsLeap[y-1900];
}

static bondsDateStruct intializeDateKernelCpu(int d, int m, int y)
{
	bondsDateStruct date;
	date.day = d;
	date.month = m;
	date.year = y;
	bool leap = isLeapCpu(y);
	int offset = monthOffsetCpu(m, leap);
	date.dateSerialNum = d + offset + yearOffsetCpu(y);
	return date;
}


// ===== extern "C" interface =====

extern "C" {

void solution_free(void)
{
	if (d_discountCurve)         { cudaFree(d_discountCurve);         d_discountCurve         = NULL; }
	if (d_repoCurve)             { cudaFree(d_repoCurve);             d_repoCurve             = NULL; }
	if (d_currDate)              { cudaFree(d_currDate);              d_currDate              = NULL; }
	if (d_maturityDate)          { cudaFree(d_maturityDate);          d_maturityDate          = NULL; }
	if (d_bondCleanPrice)        { cudaFree(d_bondCleanPrice);        d_bondCleanPrice        = NULL; }
	if (d_bond)                  { cudaFree(d_bond);                  d_bond                  = NULL; }
	if (d_dummyStrike)           { cudaFree(d_dummyStrike);           d_dummyStrike           = NULL; }
	if (d_dirtyPrice)            { cudaFree(d_dirtyPrice);            d_dirtyPrice            = NULL; }
	if (d_accruedAmountCurrDate) { cudaFree(d_accruedAmountCurrDate); d_accruedAmountCurrDate = NULL; }
	if (d_cleanPrice)            { cudaFree(d_cleanPrice);            d_cleanPrice            = NULL; }
	if (d_bondForwardVal)        { cudaFree(d_bondForwardVal);        d_bondForwardVal        = NULL; }
	g_N = 0;
}

void solution_compute(int N,
                      const int* issue_year, const int* issue_month, const int* issue_day,
                      const int* maturity_year, const int* maturity_month, const int* maturity_day,
                      const float* rates, float coupon_freq,
                      float* prices)
{
	if (g_N != N) {
		solution_free();
		// Allocate persistent GPU memory for inArgs
		cudaMalloc(&d_discountCurve, N * sizeof(bondsYieldTermStruct));
		cudaMalloc(&d_repoCurve, N * sizeof(bondsYieldTermStruct));
		cudaMalloc(&d_currDate, N * sizeof(bondsDateStruct));
		cudaMalloc(&d_maturityDate, N * sizeof(bondsDateStruct));
		cudaMalloc(&d_bondCleanPrice, N * sizeof(dataType));
		cudaMalloc(&d_bond, N * sizeof(bondStruct));
		cudaMalloc(&d_dummyStrike, N * sizeof(dataType));

		// Allocate persistent GPU memory for results
		cudaMalloc(&d_dirtyPrice, N * sizeof(dataType));
		cudaMalloc(&d_accruedAmountCurrDate, N * sizeof(dataType));
		cudaMalloc(&d_cleanPrice, N * sizeof(dataType));
		cudaMalloc(&d_bondForwardVal, N * sizeof(dataType));
		g_N = N;
	}

	// Build inArgs on host (matching bondsEngine.c / cpu_reference.c setup)
	bondsYieldTermStruct* h_discountCurve = (bondsYieldTermStruct*)malloc(N * sizeof(bondsYieldTermStruct));
	bondsYieldTermStruct* h_repoCurve = (bondsYieldTermStruct*)malloc(N * sizeof(bondsYieldTermStruct));
	bondsDateStruct* h_currDate = (bondsDateStruct*)malloc(N * sizeof(bondsDateStruct));
	bondsDateStruct* h_maturityDate = (bondsDateStruct*)malloc(N * sizeof(bondsDateStruct));
	dataType* h_bondCleanPrice = (dataType*)malloc(N * sizeof(dataType));
	bondStruct* h_bond = (bondStruct*)malloc(N * sizeof(bondStruct));
	dataType* h_dummyStrike = (dataType*)malloc(N * sizeof(dataType));

	for (int numBond = 0; numBond < N; numBond++)
	{
		dataType repoRate = 0.07;
		int repoCompounding = SIMPLE_INTEREST;
		dataType repoCompoundFreq = 1;

		bondsDateStruct bondIssueDate = intializeDateKernelCpu(issue_day[numBond], issue_month[numBond], issue_year[numBond]);
		bondsDateStruct bondMaturityDate = intializeDateKernelCpu(maturity_day[numBond], maturity_month[numBond], maturity_year[numBond]);
		bondsDateStruct todaysDate = intializeDateKernelCpu(bondMaturityDate.day-1, bondMaturityDate.month, bondMaturityDate.year);

		bondStruct bond;
		bond.startDate = bondIssueDate;
		bond.maturityDate = bondMaturityDate;
		bond.rate = rates[numBond];

		dataType bondCouponFrequency = (dataType)coupon_freq;
		dataType bondCleanPrice = 89.97693786;

		bondsYieldTermStruct bondCurve;
		bondCurve.refDate = todaysDate;
		bondCurve.calDate = todaysDate;
		bondCurve.forward = -0.1f;
		bondCurve.compounding = COMPOUNDED_INTEREST;
		bondCurve.frequency = bondCouponFrequency;
		bondCurve.dayCounter = USE_EXACT_DAY;

		dataType dummyStrike = 91.5745;

		bondsYieldTermStruct repoCurve;
		repoCurve.refDate = todaysDate;
		repoCurve.calDate = todaysDate;
		repoCurve.forward = repoRate;
		repoCurve.compounding = repoCompounding;
		repoCurve.frequency = repoCompoundFreq;
		repoCurve.dayCounter = USE_SERIAL_NUMS;

		h_discountCurve[numBond] = bondCurve;
		h_repoCurve[numBond] = repoCurve;
		h_currDate[numBond] = todaysDate;
		h_maturityDate[numBond] = bondMaturityDate;
		h_bondCleanPrice[numBond] = bondCleanPrice;
		h_bond[numBond] = bond;
		h_dummyStrike[numBond] = dummyStrike;
	}

	// Copy to device
	cudaMemcpy(d_discountCurve, h_discountCurve, N * sizeof(bondsYieldTermStruct), cudaMemcpyHostToDevice);
	cudaMemcpy(d_repoCurve, h_repoCurve, N * sizeof(bondsYieldTermStruct), cudaMemcpyHostToDevice);
	cudaMemcpy(d_currDate, h_currDate, N * sizeof(bondsDateStruct), cudaMemcpyHostToDevice);
	cudaMemcpy(d_maturityDate, h_maturityDate, N * sizeof(bondsDateStruct), cudaMemcpyHostToDevice);
	cudaMemcpy(d_bondCleanPrice, h_bondCleanPrice, N * sizeof(dataType), cudaMemcpyHostToDevice);
	cudaMemcpy(d_bond, h_bond, N * sizeof(bondStruct), cudaMemcpyHostToDevice);
	cudaMemcpy(d_dummyStrike, h_dummyStrike, N * sizeof(dataType), cudaMemcpyHostToDevice);

	free(h_discountCurve);
	free(h_repoCurve);
	free(h_currDate);
	free(h_maturityDate);
	free(h_bondCleanPrice);
	free(h_bond);
	free(h_dummyStrike);

	// Build device inArgs/results structs
	inArgsStruct inArgs;
	inArgs.discountCurve = d_discountCurve;
	inArgs.repoCurve = d_repoCurve;
	inArgs.currDate = d_currDate;
	inArgs.maturityDate = d_maturityDate;
	inArgs.bondCleanPrice = d_bondCleanPrice;
	inArgs.bond = d_bond;
	inArgs.dummyStrike = d_dummyStrike;

	resultsStruct results;
	results.dirtyPrice = d_dirtyPrice;
	results.accruedAmountCurrDate = d_accruedAmountCurrDate;
	results.cleanPrice = d_cleanPrice;
	results.bondForwardVal = d_bondForwardVal;

	// Launch kernel
	int blockSize = 256;
	int gridSize = (N + blockSize - 1) / blockSize;
	getBondsResultsGpuKernel<<<gridSize, blockSize>>>(inArgs, results, N);
	cudaDeviceSynchronize();

	// Copy results back
	dataType* h_dirtyPrice = (dataType*)malloc(N * sizeof(dataType));
	dataType* h_accruedAmountCurrDate = (dataType*)malloc(N * sizeof(dataType));
	dataType* h_cleanPrice = (dataType*)malloc(N * sizeof(dataType));
	dataType* h_bondForwardVal = (dataType*)malloc(N * sizeof(dataType));

	cudaMemcpy(h_dirtyPrice, d_dirtyPrice, N * sizeof(dataType), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_accruedAmountCurrDate, d_accruedAmountCurrDate, N * sizeof(dataType), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_cleanPrice, d_cleanPrice, N * sizeof(dataType), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_bondForwardVal, d_bondForwardVal, N * sizeof(dataType), cudaMemcpyDeviceToHost);

	for (int i = 0; i < N; i++)
	{
		prices[i * 4 + 0] = (float)h_dirtyPrice[i];
		prices[i * 4 + 1] = (float)h_accruedAmountCurrDate[i];
		prices[i * 4 + 2] = (float)h_cleanPrice[i];
		prices[i * 4 + 3] = (float)h_bondForwardVal[i];
	}

	free(h_dirtyPrice);
	free(h_accruedAmountCurrDate);
	free(h_cleanPrice);
	free(h_bondForwardVal);

	cudaDeviceSynchronize();
}

} // extern "C"

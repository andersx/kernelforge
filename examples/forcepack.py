#!/usr/bin/env python3

FORCEPACK_VERSION = 2020.1

from time import time

import numpy as np
np.set_printoptions(linewidth=666)

import scipy.stats

from tqdm import tqdm

from qml.representations import generate_fchl_acsf
from qml.math import cho_solve
from kitchen.kitchen import sample_elemental_basis
from kitchen.kitchen import get_elemental_kitchen_sinks
from kitchen.kitchen import get_elemental_kitchen_sink_gradient
from kitchen.kitchen import train_elemental_kitchen_sinks
from kitchen.kitchen import train_elemental_kitchen_sink_gradient

from copy import deepcopy

import gc

def calc_mae(A, B):
    return np.mean(np.abs(np.asarray(A).flatten()-np.asarray(B).flatten()))

def calc_rmse(A, B):
    return np.sqrt(np.mean(np.square(np.asarray(A).flatten()-np.asarray(B).flatten())))

def calc_pearsonr(A, B):
    (r, p) = scipy.stats.pearsonr(np.asarray(A).flatten(), np.asarray(B).flatten())
    return r

class EnergyTrainedModel:


    def __init__(self, nbasis=None, verbose=True):

        if verbose:
            print(f"ForcePack \U0001F3CB  {FORCEPACK_VERSION} written by Anders S. Christensen")

        self.nbasis = nbasis

        self._initialized = False

        self.R_training = None
        self.Z_training = None

        self.reductor = None


    def apply_feature_reduction(self, nreps=500, npcas=200, verbose=False):

        n_mols = len(self.Z_training)

        idx_permutation = np.random.permutation(n_mols)

        svd_reps = dict()
        for a in self.elements:
            svd_reps[a] = []

        for i, idx in tqdm(enumerate(idx_permutation), disable=not verbose):

            if all([len(svd_reps[a]) >= nreps for a in self.elements]):
                break

            rep = generate_fchl_acsf(self.Z_training[idx], self.R_training[idx],
                gradients=False,
                elements=self.elements,
                **self.rep_kwargs)

            for a in self.elements:
                if len(svd_reps[a]) >= nreps: continue
                if a not in self.Z_training[idx]: continue
                aidx = np.where(np.asarray(self.Z_training[idx]) == a)[0]
                selection = np.random.choice(aidx)

                svd_reps[a].append(rep[selection])

        reductor = dict()

        for a in self.elements:

            svd_reps[a] = np.array(svd_reps[a])

            eigen_vecs, eigen_vals, Vh = np.linalg.svd(svd_reps[a].T,
                full_matrices=False, compute_uv=True)

            cev = 100 - (np.sum(eigen_vals) - np.sum(eigen_vals[:npcas]))  / np.sum(eigen_vals) * 100

            reductor[a] = eigen_vecs[:,:npcas]

            if verbose:
                size_from = reductor[a].shape[0]
                size_to = reductor[a].shape[1]
                print(f"Element {a:3d}  {size_from} -> {size_to}  Cumulative Explained Feature Variance = {cev:6.2f} %%")

        self.reductor = reductor
        self.npcas = npcas

        return reductor


    def get_reps(self, R_input, Z_input, verbose=False):

        padmax = max([len(_) for _ in Z_input])

        X  = []

        for i, (this_z, this_r) in enumerate(tqdm(list(zip(Z_input, R_input)), disable=not verbose, desc="Generating representations")):
            # print(this_r.shape, padmax)
            rep = generate_fchl_acsf(Z_input[i], R_input[i],
                gradients=False,
                pad=padmax,
                elements=self.elements,
                **self.rep_kwargs)

            if self.reductor is not None:
                reduced_rep = np.zeros((padmax,self.npcas))

                for j, a in enumerate(this_z):

                    reduced_rep[j] = rep[j] @ self.reductor[a]

                X.append(reduced_rep)
            else:
                X.append(rep)

        X = np.asarray(X)

        return X


    def get_reps_derivatives(self, R_input, Z_input, verbose=False):

        padmax = max([len(_) for _ in Z_input])

        X = []
        dX = []

        for i, (this_z, this_r) in enumerate(tqdm(list(zip(Z_input, R_input)), disable=not verbose, desc="Generating representations")):
            rep, drep = generate_fchl_acsf(this_z, this_r,
                    gradients=True,
                    pad=padmax,
                    elements=self.elements,
                    **self.rep_kwargs
                )

            if self.reductor is not None:
                reduced_rep = np.zeros((padmax,self.npcas))
                reduced_drep = np.zeros((padmax,self.npcas,padmax,3))

                for j, a in enumerate(this_z):
                    reduced_rep[j] = rep[j] @ self.reductor[a]
                    reduced_drep[j] = np.einsum("kmn,kl->lmn", drep[j], self.reductor[a])

                X.append(reduced_rep)
                dX.append(reduced_drep)
            else:
                X.append(rep)
                dX.append(drep)

        X = np.asarray(X)
        dX = np.asarray(dX)

        return X, dX


    def set_training(self, R=None, Z=None, E=None):

        assert len(R) == len(Z)
        assert len(R) == len(E)

        self.R_training = R
        self.Z_training = Z
        self.E_training = np.asarray(E).flatten()

        self.elements = np.unique(self.Z_training)

        # print(self.elements)

        self.training_points = len(E)

        # print(len(E))

        return


    def set_parameters(self, sigma=2.0, llambda=1e-8, nbasis=None, rep_kwargs={}, verbose=False):

        self.nbasis = nbasis
        self.sigma = sigma
        self.llambda = llambda
        self.rep_kwargs = rep_kwargs

        if verbose:
            print(f"Setting parameters:")
            print(f"-------------------------------------------------")
            if self.nbasis is not None:
                print(f"Basis functions (D)        nbasis      {self.nbasis:10d}")
            if self.llambda is not None:
                print(f"L2-regularization (\u03BB)      llambda     {self.llambda:10.2e}")
            if self.sigma is not None:
                print(f"Kernel width (\u03C3)           sigma       {self.sigma:10.2f}")
            if len(self.rep_kwargs) > 0:
                print(f"Representation rep_kwargs:", rep_kwargs)


    def train(self, idx=None, n=None, batch_size=10000, verbose=True):

        if verbose:
            print("Starting energy training ...")

        if idx is None and n is None:
            idx = np.arange(0, self.training_points)

        elif idx is None and n is not None:
            idx = sorted(np.random.choice(self.training_points, size=n, replace=False))

        else:
            idx = np.asarray(idx)

        if self.nbasis is None:
            self.nbasis = len(idx)

        D = self.nbasis

        X = self.get_reps(
                [self.R_training[i] for i in idx],
                [self.Z_training[i] for i in idx],
                verbose=verbose,
            )

        if verbose:
            print("Sampling Fourier basis ...")

        self.W, self.b = sample_elemental_basis(
                X,
                sigma=self.sigma,
                size=D,
                elements=self.elements
            )

        Q = [self.Z_training[i] for i in idx]

        # self.offset = np.mean(self.E_training[idx])
        E_train = deepcopy(self.E_training[idx])# - self.offset

        if verbose:
            print("Calculating Gramian matrix iteratively ...")


        LZTLZ, LZTY = train_elemental_kitchen_sinks(
            X,
            Q,
            self.W,
            self.b,
            E_train,
            chunk_size=batch_size,
        )

        if verbose:
            print("Cholesky solver ...")

        self.alpha = cho_solve(LZTLZ, LZTY, l2reg=self.llambda, destructive=True)


    def predict(self, R=None, Z=None):

        assert R is not None
        assert Z is not None

        X = np.asfortranarray(self.get_reps(R, Z))

        LZ = get_elemental_kitchen_sinks(X, Z, self.W, self.b)

        E = np.dot(LZ, self.alpha) # + self.offset

        return E


    def predict_forces(self, R=None, Z=None, verbose=False):

        if verbose:
            print("Starting force+energy prediction ...")

        assert R is not None
        assert Z is not None

        if verbose:
            print("Generating representations+derivatives ...")

        X, dX = self.get_reps_derivatives(R, Z)
        X = np.asfortranarray(X)
        dX = np.asfortranarray(dX)

        if verbose:
            print("Calculating energies ...")

        LZ = get_elemental_kitchen_sinks(X, Z, self.W, self.b)
        E = np.dot(LZ, self.alpha) # + self.offset

        LZ = None; del LZ

        if verbose:
            print("Calculating forces ...")

        dLZ = get_elemental_kitchen_sink_gradient(X, dX, Z, self.W, self.b)
        F = np.dot(dLZ.T, self.alpha)

        return E, F


    def nested_grid_cv(self, idx=None, n=None, nbasis=None, sigmas=None, llambdas=None, verbose=None, batch_size=10000):

        if sigmas is None:
            sigmas = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]

        if llambdas is None:
            llambdas = [1e-12, 1e-11, 1e-10, 1e-09, 1e-08, 1e-07, 1e-06, 1e-05]

        if idx is None and n is None:
            idx = np.arange(0, self.training_points)

        elif idx is None and n is not None:
            idx = sorted(np.random.choice(self.training_points, size=n, replace=False))

        else:
            idx = np.asarray(idx)

        if self.nbasis is None:
            self.nbasis = len(idx)


        if nbasis is None:
            nbasis = len(idx)

        if verbose:
            print("Generating representations ...")

        Xall = self.get_reps(
                [self.R_training[i] for i in idx],
                [self.Z_training[i] for i in idx],
                verbose=verbose,
            )

        Qall = [self.Z_training[i] for i in idx]

        from sklearn.model_selection import KFold

        kf = KFold(n_splits=4, shuffle=True)
        splits = [s for s in kf.split(list(range(len(idx))))]

        all_rmse = dict()
        all_mae = dict()
        all_pearsonr = dict()

        Yall = self.E_training[idx]

        for fold, (train, test) in enumerate(splits):

            X = np.asfortranarray(Xall[train])
            Xs = np.asfortranarray(Xall[test])

            Y = Yall[train]
            Ys = Yall[test]

            Q  = [Qall[i] for i in train]
            Qs = [Qall[i] for i in test]

            D = int(np.rint(len(train) / n * nbasis))

            for sigma in sigmas:

                t1 = time()

                self.W, self.b = sample_elemental_basis(
                    X,
                    sigma=sigma,
                    size=D,
                    elements=self.elements
                )

                t2 = time()

                LZTLZ, LZTY = train_elemental_kitchen_sinks(
                    X,
                    Q,
                    self.W,
                    self.b,
                    Y,
                    chunk_size=batch_size,
                )

                LZTLZ = np.ascontiguousarray(LZTLZ)


                alphas = []

                for llambda in llambdas:

                    alpha = cho_solve(LZTLZ, LZTY, l2reg=llambda, destructive=True)
                    alphas.append(alpha)

                LZTLZ = None; del LZTLZ

                t3 = time()
                LZs = get_elemental_kitchen_sinks(Xs, Qs, self.W, self.b)
                t4 = time()

                if verbose:
                    print(f"Sigma {sigma}  N {len(train)}  Basis {t2-t1:.2f}   Train {t3-t2:.2f} s  Test {t4-t3:.2f} s")

                for l, llambda in enumerate(llambdas):

                    Yss = np.dot(LZs, alphas[l])

                    rmse = calc_rmse(Ys, Yss)
                    mae  = calc_mae(Ys, Yss)
                    pearsonr = calc_pearsonr(Ys, Yss)

                    score = (mae, rmse, pearsonr)
                    key = (sigma, llambda)

                    if key not in all_rmse.keys(): all_rmse[key] = []
                    if key not in all_mae.keys(): all_mae[key] = []
                    if key not in all_pearsonr.keys(): all_pearsonr[key] = []

                    all_rmse[key].append(rmse)
                    all_mae[key].append(mae)
                    all_pearsonr[key].append(pearsonr)

                    if verbose:
                        print(f"Inner CV: {fold:2d}  sigma = {sigma:6.2f}  lambda = {llambda:6.2E}  MAE = {mae:10.4f}  RMSE = {rmse:10.4f}")

                LZs = None; del LZs

            X = None; del X
            Xs = None; del Xs

        return {"pearsonr": all_pearsonr, "mae": all_mae, "rmse": all_rmse}


class ForceTrainedModel(EnergyTrainedModel):


    def __init__(self, nbasis=None, verbose=False):

        super().__init__(nbasis=nbasis, verbose=verbose)

        print("Initialized ForceTrainedModel!")

    def set_training(self, R=None, Z=None, E=None, F=None):

        assert len(R) == len(Z)
        assert len(R) == len(E)

        self.R_training = R
        self.Z_training = Z
        self.E_training = E
        self.F_training = F

        self.elements = np.unique(self.Z_training)
        self.E_training = np.asarray(E).flatten()

        # print(self.elements)

        self.training_points = len(E)

        # self.padmax = max([len(self.Z_training[i]) for i in idx])
        # print(len(E))

        return


    def train(self, idx=None, n=None, batch_size=10000, verbose=False):


        if verbose:
            print("Starting force+energy training ...")

        if idx is None and n is None:
            idx = np.arange(0, self.training_points)

        elif idx is None and n is not None:
            idx = sorted(np.random.choice(self.training_points, size=n, replace=False))

        else:
            idx = np.asarray(idx)

        # self.offset = np.mean(self.E_training[idx])
        # E_train = self.E_training[idx]# - self.offset
        E_train = deepcopy(self.E_training[idx])# - self.offset

        F_train = np.concatenate([np.asarray(self.F_training[i]).flatten() for i in idx])#.flatten()

        if self.nbasis is None:
            self.nbasis = len(E_train) + len(F_train)

        X, dX = self.get_reps_derivatives(
                [self.R_training[i] for i in idx],
                [self.Z_training[i] for i in idx],
                verbose=verbose,
            )

        X = np.asfortranarray(X)
        dX = np.asfortranarray(dX)

        if verbose:

            print(f"Model Summary:")
            print(f"-----------------------------------")
            print(f"Basis functions:         {self.nbasis:7d}")
            print(f"Energy labels:           {len(E_train):7d}")
            print(f"Force labels:            {len(F_train):7d}")
            gramian_size = self.nbasis**2  * 8 / (1000**3)
            print(f"Mem: Gramian       {gramian_size:10.3} GB")
            rep_size = X.shape[0] * X.shape[1] *X.shape[2] * 8 / (1000**3)
            drep_size = dX.shape[0] * dX.shape[1] * dX.shape[2] * dX.shape[3] * dX.shape[4]* 8 / (1000**3)
            print(f"Mem: Reps          {rep_size:10.3} GB")
            print(f"Mem: Deriv reps    {drep_size:10.3} GB")
            print()

        self.W, self.b = sample_elemental_basis(
                X,
                sigma=self.sigma,
                size=self.nbasis,
                elements=self.elements
            )

        Q = [self.Z_training[i] for i in idx]

        if verbose:
            print(f"Calculating Gramian matrix: G = (LZ)ᵀ(LZ) + (d/dR LZ)ᵀ(d/dR LZ) ...\n")

        GZTGZ, GZTY = train_elemental_kitchen_sink_gradient(
                X,
                dX,
                Q,
                self.W,
                self.b,
                E_train,
                F_train,
                chunk_size=batch_size,
            )

        if verbose:
            print(f"Cholesky-decomposition to solve α = (G+Iλ)⁻¹((LZ)ᵀE + (d/dR LZ)ᵀF) ...\n")
        self.alpha = cho_solve(GZTGZ, GZTY, l2reg=self.llambda, destructive=True)


    def nested_grid_cv(self, idx=None, n=None, nbasis=None, sigmas=None, llambdas=None, verbose=None, batch_size=10000):

        if sigmas is None:
            sigmas = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]

        if llambdas is None:
            llambdas = [1e-12, 1e-11, 1e-10, 1e-09, 1e-08, 1e-07, 1e-06, 1e-05]

        if idx is None and n is None:
            idx = np.arange(0, self.training_points)

        elif idx is None and n is not None:
            idx = sorted(np.random.choice(self.training_points, size=n, replace=False))

        else:
            idx = np.asarray(idx)

        E_train = deepcopy(self.E_training[idx])# - self.offset
        F_train = np.concatenate([np.asarray(self.F_training[i]).flatten() for i in idx])#.flatten()

        if nbasis is None:
            nbasis = len(E_train) + len(F_train)

        Xall, dXall = self.get_reps_derivatives(
                            [self.R_training[i] for i in idx],
                            [self.Z_training[i] for i in idx],
                            verbose=verbose,
                        )

        Qall = [self.Z_training[i] for i in idx]

        from sklearn.model_selection import KFold

        kf = KFold(n_splits=4, shuffle=True)
        splits = [s for s in kf.split(list(range(len(idx))))]

        all_rmse = dict()
        all_mae = dict()
        all_pearsonr = dict()

        Eall = deepcopy(self.E_training[idx])# - self.offset
        Fall = [np.asarray(self.F_training[i]) for i in idx]

        Yall = np.concatenate((
                Eall,
                np.concatenate(Fall).flatten(),
            ))



        for fold, (train, test) in enumerate(splits):

            X = np.asfortranarray(Xall[train])
            Xs = np.asfortranarray(Xall[test])

            dX = np.asfortranarray(dXall[train])
            dXs = np.asfortranarray(dXall[test])

            eY = Eall[train]
            eYs = Eall[test]

            fY = np.concatenate([Fall[i] for i in train]).flatten()
            fYs = np.concatenate([Fall[i] for i in test]).flatten()

            Y = np.concatenate((eY, fY))

            Q  = [Qall[i] for i in train]
            Qs = [Qall[i] for i in test]

            D = int(np.rint(len(Y) / len(Yall) * nbasis))

            for sigma in sigmas:

                t1 = time()

                self.W, self.b = sample_elemental_basis(
                    X,
                    sigma=sigma,
                    size=D,
                    elements=self.elements
                )

                t2 = time()

                alphas = []

                GZTGZ, GZTY = train_elemental_kitchen_sink_gradient(
                        X,
                        dX,
                        Q,
                        self.W,
                        self.b,
                        eY, # E_train,
                        fY, # F_train,
                        chunk_size=batch_size,
                        )
                GZTGZ = np.ascontiguousarray(GZTGZ)
                t3 = time()

                for llambda in tqdm(llambdas, desc=f"Solving for \u03B1", disable=not verbose):
                    alpha = cho_solve(GZTGZ, GZTY, l2reg=llambda, destructive=True)
                    alphas.append(alpha)

                t4 = time()

                GZTGZ = None; del GZTGZ
                LZs = get_elemental_kitchen_sinks(Xs, Qs, self.W, self.b)
                dLZ = get_elemental_kitchen_sink_gradient(Xs, dXs, Qs, self.W, self.b)

                t5 = time()

                if verbose:
                    print(f"Summary:  \u03C3 = {sigma:6.2f}  Split: {len(train)} / {len(test)}  Basis: {t2-t1:.2f} s   Gramian: {t3-t2:.2f} s  Cholesky: {t4-t3:.2f}   Test: {t5-t4:.2f} s")

                for l, llambda in enumerate(llambdas):


                    # alpha = cho_solve(GZTGZ, GZTY, l2reg=llambda, destructive=True)

                    alpha = alphas[l]

                    eYss = np.dot(LZs, alpha)
                    fYss = np.dot(dLZ.T, alpha)

                    ermse = calc_rmse(eYs, eYss)
                    emae  = calc_mae(eYs, eYss)
                    epearsonr = calc_pearsonr(eYs, eYss)

                    frmse = calc_rmse(fYs, fYss)
                    fmae  = calc_mae(fYs, fYss)
                    fpearsonr = calc_pearsonr(fYs, fYss)

                    # score = (mae, rmse, pearsonr)
                    key = (sigma, llambda)

                    if key not in all_rmse.keys(): all_rmse[key] = []
                    if key not in all_mae.keys(): all_mae[key] = []
                    if key not in all_pearsonr.keys(): all_pearsonr[key] = []

                    all_rmse[key].append((ermse,frmse))
                    all_mae[key].append((emae, fmae))
                    all_pearsonr[key].append((epearsonr,fpearsonr))

                    if verbose:
                        print(f"Inner CV: {fold:2d}  \u03C3 = {sigma:6.2f}  \u03BB = {llambda:6.2e}  MAE(E) = {emae:10.4f}  RMSE(E) = {ermse:10.4f}  MAE(F) = {fmae:10.4f}  RMSE(F) = {frmse:10.4f}")
                LZs = None; del LZs
                dLZs = None; del dLZs

            X = None; del X
            dX = None; del dX

            Xs = None; del Xs
            dXs = None; del dXs


        return {"pearsonr": all_pearsonr, "mae": all_mae, "rmse": all_rmse}

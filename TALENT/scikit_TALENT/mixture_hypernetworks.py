
from collections import defaultdict

import numpy as np

from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import KMeans
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score, pairwise_distances
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample

from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion

from ticl.prediction import MotherNetClassifier

#--------------------------------
# Clustering 

class BalancedKMeansLSA_optimized(BaseEstimator, ClusterMixin):
    """
    KMeans + (balanced / batched) Linear Sum Assignment (LSA).

    Key guarantee:
    -------------
    If N <= lsa_batch_threshold, this matches BalancedKMeansLSA EXACTLY:
    - capacities sum to N (base + remainder), and cost matrix is (N x N).

    For N > lsa_batch_threshold:
    - Uses your fixed+overflow + batched LSA logic.
    """

    def __init__(
        self,
        n_clusters=8,
        max_iter=300,
        n_init=10,
        random_state=None,
        lsa_batch_threshold=5000,
        batch_size=5000,
        capacity_multiplier=4.0,  # used ONLY when N > threshold
    ):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_init = n_init
        self.random_state = random_state
        self.lsa_batch_threshold = lsa_batch_threshold
        self.batch_size = batch_size
        self.capacity_multiplier = capacity_multiplier

    # ---------- helpers ----------

    def _build_slots(self, clusters, caps):
        slot_clusters = []
        for c, cap in zip(clusters, caps):
            if cap > 0:
                slot_clusters.extend([c] * int(cap))
        return np.asarray(slot_clusters, dtype=int)

    def _lsa_assign(self, D_sub, slot_clusters):
        # D_sub: (m, k); slot_clusters: (n_slots,)
        cost = D_sub[:, slot_clusters]  # (m, n_slots) with repeated centroid columns
        row_ind, col_ind = linear_sum_assignment(cost)
        labels_sub = np.empty(D_sub.shape[0], dtype=int)
        labels_sub[row_ind] = slot_clusters[col_ind]
        return labels_sub

    def _batched_lsa(self, D, remaining_idx, cap_rem, batch_size):
        """
        Assign remaining_idx using batched LSA under remaining capacities cap_rem.
        D: (N,k) squared distances for all points
        remaining_idx: indices to assign
        cap_rem: (k,) remaining capacities per cluster
        """
        N, k = D.shape
        labels_out = np.full(N, -1, dtype=int)

        rem = remaining_idx.copy()
        # assign "easy" points first
        min_d = D[rem].min(axis=1)
        rem = rem[np.argsort(min_d)]

        while rem.size > 0:
            m = min(int(batch_size), rem.size)
            batch = rem[:m]
            rem = rem[m:]

            active = np.where(cap_rem > 0)[0]
            if active.size == 0:
                labels_out[batch] = D[batch].argmin(axis=1)
                continue

            total_cap = int(cap_rem[active].sum())

            raw = (cap_rem[active] / total_cap) * m
            batch_caps = np.floor(raw).astype(int)

            # avoid starving
            for i, c in enumerate(active):
                if cap_rem[c] > 0 and batch_caps[i] == 0:
                    batch_caps[i] = 1
            batch_caps = np.minimum(batch_caps, cap_rem[active])

            # top up if short
            slots = int(batch_caps.sum())
            if slots < m:
                need = m - slots
                order = np.argsort(-cap_rem[active])
                for idx in order:
                    if need <= 0:
                        break
                    c = active[idx]
                    addable = int(cap_rem[c] - batch_caps[idx])
                    if addable <= 0:
                        continue
                    add = min(addable, need)
                    batch_caps[idx] += add
                    need -= add

            # if still short, do LSA for what we can; leftover nearest
            if int(batch_caps.sum()) < m:
                m2 = int(batch_caps.sum())
                if m2 <= 0:
                    labels_out[batch] = D[batch].argmin(axis=1)
                    continue
                batch2 = batch[:m2]
                batch_rest = batch[m2:]
            else:
                batch2 = batch
                batch_rest = None

            slot_clusters = self._build_slots(active.tolist(), batch_caps.tolist())
            labels_b = self._lsa_assign(D[batch2], slot_clusters)
            labels_out[batch2] = labels_b

            used = np.bincount(labels_b, minlength=k)
            cap_rem = cap_rem - used

            if batch_rest is not None and batch_rest.size > 0:
                labels_out[batch_rest] = D[batch_rest].argmin(axis=1)

        return labels_out

    # ---------- main API ----------

    def fit(self, X, y=None):
        X = np.asarray(X)
        N, n_features = X.shape
        self.n_features_in_ = n_features

        if self.n_clusters <= 0:
            raise ValueError("n_clusters must be a positive integer.")

        k = min(self.n_clusters, N)
        self.n_clusters_ = k

        if k == 1:
            self.labels_ = np.zeros(N, dtype=int)
            self.cluster_centers_ = X.mean(axis=0, keepdims=True)
            return self

        # 1) Initial KMeans
        self.kmeans_ = KMeans(
            n_clusters=k,
            max_iter=self.max_iter,
            n_init=self.n_init,
            random_state=self.random_state,
        )
        init_labels = self.kmeans_.fit_predict(X)
        centroids = self.kmeans_.cluster_centers_

        # 2) Distances
        D = cdist(X, centroids, metric="sqeuclidean")  # (N,k)

        # ------------------------------------------------------------
        # Small-N: EXACTLY match BalancedKMeansLSA
        # ------------------------------------------------------------
        if N <= int(self.lsa_batch_threshold):
            base = N // k
            remainder = N % k
            capacities = np.array(
                [base + (1 if j < remainder else 0) for j in range(k)],
                dtype=int,
            )
            # capacities sum to N; each >= 1 because k <= N

            clusters = np.arange(k, dtype=int)
            slot_clusters = self._build_slots(clusters.tolist(), capacities.tolist())
            # slot_clusters length == N (critical to match original)

            labels = self._lsa_assign(D, slot_clusters)
            self.labels_ = labels

            print("BalancedKMeansLSA: Cluster sizes:", np.bincount(self.labels_, minlength=k))

            centers = []
            for j in range(k):
                mask = self.labels_ == j
                centers.append(X[mask].mean(axis=0))  # no empties by construction
            self.cluster_centers_ = np.vstack(centers)
            return self

        # ------------------------------------------------------------
        # Large-N: your fixed+overflow + batched LSA path
        # ------------------------------------------------------------
        cap_base = int(np.ceil(N / k))

        labels = np.full(N, -1, dtype=int)
        fixed = np.zeros(N, dtype=bool)
        fixed_count = np.zeros(k, dtype=int)

        # Fix exactly min(size_j, cap_base) closest-to-own-centroid in each KMeans cluster
        for j in range(k):
            idx_j = np.where(init_labels == j)[0]
            s = idx_j.size
            if s == 0:
                continue

            n_fix = min(s, cap_base)
            dj = D[idx_j, j]
            keep_local = np.argsort(dj)[:n_fix]
            keep_idx = idx_j[keep_local]

            labels[keep_idx] = j
            fixed[keep_idx] = True
            fixed_count[j] = keep_idx.size

        remaining_idx = np.where(~fixed)[0]
        n_remaining = remaining_idx.size

        cap_target = int(np.ceil(cap_base * float(self.capacity_multiplier)))
        cap_rem = np.maximum(0, cap_target - fixed_count)

        if int(cap_rem.sum()) < int(n_remaining):
            deficit = int(n_remaining - cap_rem.sum())
            for t in range(deficit):
                cap_rem[t % k] += 1

        if n_remaining > 0:
            labels_rem = self._batched_lsa(D, remaining_idx, cap_rem.copy(), int(self.batch_size))
            labels[remaining_idx] = labels_rem[remaining_idx]

        self.labels_ = labels

        centers = []
        for j in range(k):
            mask = self.labels_ == j
            centers.append(X[mask].mean(axis=0) if np.any(mask) else centroids[j])
        self.cluster_centers_ = np.vstack(centers)

        print("BalancedKMeansLSA: Cluster sizes:", np.bincount(self.labels_, minlength=k))
        return self

    def fit_predict(self, X, y=None):
        return self.fit(X, y).labels_

    def predict(self, X):
        X = np.asarray(X)
        return pairwise_distances_argmin(X, self.cluster_centers_)


#--------------------------------
# Dual-MoH

class Dual_MoH():
    def __init__(self, n_classifiers=2, overlap=0.25, random_state = None, verbose=False, m=3, alpha = 0.5, majority_cluster="KMeans",minority_cluster="KMeans",classifier_type='MotherNet'):
        self.m = m
        self.n_classifiers = n_classifiers
        self.verbose = verbose
        self.overlap_thr = overlap
        self.random_state = random_state
        self.majority_cluster = majority_cluster
        self.minority_cluster = minority_cluster
        self.classifier_type = classifier_type
        self.alpha = alpha
        
    def compute_classwise_stats(self, y):# Get unique values and their counts
        unique_vals, counts = np.unique(y, return_counts=True)

        # Sort by frequency (descending) and assign major/minor classes
        sorted_indices = np.argsort(-counts)
        class_counts = {
            "maj": {"class": unique_vals[sorted_indices[0]], "count": counts[sorted_indices[0]]},
            "min": {"class": unique_vals[sorted_indices[1]], "count": counts[sorted_indices[1]]},
        }
        self.class_counts_ = class_counts

    def compute_classwise_cluster(self, X,y):

        
        # compute number of clusters per class
        n_clusters_maj = np.ceil(self.class_counts_["maj"]["count"]**(1/3))
        n_clusters_min = np.ceil(n_clusters_maj*(self.class_counts_["min"]["count"]/self.class_counts_["maj"]["count"])*self.m)
        
        if n_clusters_min > n_clusters_maj:
            n_clusters_min = n_clusters_maj
            
            print(f"Adjusted n_clusters_min to n_clusters_maj: {n_clusters_min}")
            
        #ensure at least 1 cluster in minority class
        n_clusters_min = max(1, n_clusters_min)
        
        
        if self.verbose:
            print(f"Class distribution - min class: {self.class_counts_['min']['count']}, maj class: {self.class_counts_['maj']['count']}")
            print(f"Number of clusters - min class: {n_clusters_min}, maj class: {n_clusters_maj} | self.m: {self.m} | ratio: {self.class_counts_['min']['count']//self.class_counts_['maj']['count']}")
        
        #avoid mistakes from n_cluster_min being larger than minority samples and throw a warning
        if n_clusters_min > self.class_counts_["min"]["count"]:
            print(f"Warning: n_clusters_min ({n_clusters_min}) is larger than number of minority samples ({self.class_counts_['min']['count']}). Setting n_clusters_min to {self.class_counts_['min']['count']}.")
            n_clusters_min = min(n_clusters_min, self.class_counts_["min"]["count"])
        
        #fit kmeans per class
        
        #min class
        X_min = X[y==self.class_counts_["min"]["class"]]
        X_maj = X[y==self.class_counts_["maj"]["class"]]
        
        if self.minority_cluster == "KMeans":
            if X_min.shape[0] < 5000:
                kmeans_min = KMeans(n_clusters=int(n_clusters_min),random_state=self.random_state)
            # else:
            #     kmeans_min = BalancedKMeansLSA_optimized_v2(n_clusters=int(n_clusters_min),random_state=self.random_state)
        elif self.minority_cluster == "BalancedKMeansLSA":
            kmeans_min = BalancedKMeansLSA_optimized(n_clusters=int(n_clusters_min),random_state=self.random_state)
        else:
            raise ValueError(f"Unknown minority clustering method: {self.minority_cluster}")
        
        kmeans_min.fit(X_min)
        self.min_clusters  = kmeans_min.labels_
        self.min_centroids = kmeans_min.cluster_centers_
        self.counters_per_min_cluster = np.bincount(self.min_clusters)
        
        #maj class
        if self.majority_cluster == "KMeans":
            kmeans_maj = KMeans(n_clusters=int(n_clusters_maj),random_state=self.random_state)
        elif self.majority_cluster == "BalancedKMeansLSA":
            kmeans_maj = BalancedKMeansLSA_optimized(n_clusters=int(n_clusters_maj),random_state=self.random_state)
        else:
            raise ValueError(f"Unknown majority clustering method: {self.majority_cluster}")

        kmeans_maj.fit(X_maj)
        self.maj_clusters  = kmeans_maj.labels_ 
        self.maj_centroids = kmeans_maj.cluster_centers_
        self.counts_per_maj_cluster = np.bincount(self.maj_clusters)
        
        print(f"cluster distribution - min clusters: {self.counters_per_min_cluster}, maj clusters: {self.counts_per_maj_cluster}")

    def get_subdomains(self):
        
        #NOTE: self. domains final contains randomly sampled majority centroids equal to number of minority samples
        
        # compute pairwise distances between majority centroids and minority centroids
        distances = pairwise_distances(self.maj_centroids,self.min_centroids)
        
        if self.verbose:
            print("Pairwise distances shape:", distances.shape) 
        
        #get closest m majority centroids for each minority centroid
        sorted_indices_upto_m = np.argsort(distances, axis=1)
        
        #generate subdomains (first value is majority centroid index, rest are minority centroid indices)
        self.subdomains = []
        self.subdomains_centroids = []
        self.domains_final = []
        self.centroids_final = []
        
        for maj_idx, min_indices in enumerate(sorted_indices_upto_m):
            
            if self.verbose:
                print(f"counters_per_min_cluster: {self.counters_per_min_cluster.shape} len(min_indices): {len(min_indices)}")
            
            cumulative_counts = np.cumsum(self.counters_per_min_cluster[min_indices])
            
            if self.verbose:
                print(f"Cumulative counts last: {cumulative_counts[-1]} | Majority cluster {maj_idx} requires {self.counts_per_maj_cluster[maj_idx]} samples.")
            
            thr = int(np.round(self.counts_per_maj_cluster[maj_idx] * (1 + self.overlap_thr)))
            
            #if cumulative counts[-1] is not enough to cover majority cluster, take all minority clusters
            if cumulative_counts[-1] < thr:
                thr = len(cumulative_counts) -1
                min_indices_right = min_indices
                
            else:
                thr = np.where(cumulative_counts >= thr)[0]
                min_indices_right = min_indices[:thr[0]+1]

                if self.verbose:
                    print(f"Cumulative samples are thr: {cumulative_counts[thr[0]]} | Majority cluster {maj_idx} requires {self.counts_per_maj_cluster[maj_idx]} samples | number of selected clusters: {len(min_indices_right)}")
            
            subdomain = [maj_idx] + min_indices_right.tolist()
            subdomain_centroids = [self.maj_centroids[maj_idx]] 
            self.subdomains.append(subdomain)
            self.subdomains_centroids.append(np.mean(subdomain_centroids,axis=0))
              
        #print("Subdomains centroids shape:", np.array(subdomain_centroids).shape,np.array(self.subdomains_centroids).shape)    

    def train_meta_models(self,X):

        # train meta models to predict the best model for each subdomain based on self.domain_indices_on_X
        self.meta_models_balanced_ = {}
        self.meta_models_empirical_ = {}
        
        # iterate over self.domain_indices_on_X to train a meta model for each subdomain
        for i in range(len(self.domain_indices_on_X)):
            y_meta_balanced = self.domain_indices_on_X[i]

            model = DecisionTreeClassifier(random_state=42,class_weight='balanced')
            #model = MotherNetClassifier(device='cuda')
            #model = RandomForestClassifier(n_estimators=10, random_state=42, class_weight='balanced_subsample')
            model.fit(X, y_meta_balanced)
            self.meta_models_balanced_[f"Meta_Model_{i}"] = model
            
            # empirical
            if i < len(self.domain_indices_on_X_empirical):
                y_meta_empirical = self.domain_indices_on_X_empirical[i]
                
                model = DecisionTreeClassifier(random_state=42,class_weight='balanced')
                #model = MotherNetClassifier(device='cuda')
                #model = RandomForestClassifier(n_estimators=10, random_state=42, class_weight='balanced_subsample')
                model.fit(X, y_meta_empirical)
                self.meta_models_empirical_[f"Meta_Model_{i}"] = model
                if self.verbose:
                    print(f"Empirical: Trained meta model for subdomain {i} with {np.sum(y_meta_empirical)} positive samples.")

            if self.verbose:
                print(f"Balanced: Trained meta model for subdomain {i} with {np.sum(y_meta_balanced)} positive samples.")
    

    def generate_empirical_subdomains(self,subdomain_empirical, X, y):
        
        #add minority cluster
        idx_min  = np.flatnonzero(y == self.class_counts_["min"]["class"])
        idx_max  = np.flatnonzero(y == self.class_counts_["maj"]["class"])
        
        indexes = self.min_clusters == subdomain_empirical[0]
        
        X_min_sub =  X[idx_min[indexes]]
        
        #track empirical domain indices
        bolean = np.zeros((len(y),))   
        bolean[idx_min[indexes]] = 1
        
        maj_size = 0
        #add majority samples
        for maj_cluster in subdomain_empirical[1:]:
            indexes_maj = self.maj_clusters == maj_cluster
            X_maj_sub = X[idx_max[indexes_maj]]
            X_min_sub = np.vstack([X_min_sub, X_maj_sub])
            maj_size += X_maj_sub.shape[0]
            
            #track empirical domain indices
            bolean[idx_max[indexes_maj]] = 1
            
        #generate labels
        y_sub = np.hstack([np.full(X_min_sub.shape[0]-maj_size, self.class_counts_["min"]["class"]),
                           np.full(maj_size, self.class_counts_["maj"]["class"])])  
        
        if self.verbose:
            print(f"Empirical subdomain shape: {X_min_sub.shape}, class distribution: {np.unique(y_sub,return_counts=True)}")
        
        return X_min_sub, y_sub, bolean
                
    def fit_empirical_mixture_hypernetworks(self, X, y):
        
        # the algorith is, for each minority subdomain, select the closest majority cluster without replacement
        # then unselected majority clusters not selected and assigned them to the closest subdomain until all majority clusters are assigned
        # compute centroids of all subdomains via mean of all samples in the subdomain. After subdomain formation, fit MotherNetClassifier on each subdomain
        distances = pairwise_distances(self.min_centroids,self.maj_centroids)
        sorted_maj_indices = np.argsort(distances, axis=1)
        
        # assign closest majority cluster to each minority subdomain without replacement
        assigned_maj_clusters = set()
        subdomains_emp = []
        
        for min_idx in range(self.min_centroids.shape[0]):
            for maj_candidate in sorted_maj_indices[min_idx]:
                if maj_candidate not in assigned_maj_clusters:
                    assigned_maj_clusters.add(maj_candidate)
                    subdomain = [min_idx,maj_candidate]
                    subdomains_emp.append(subdomain)
                    break
                
        
        # assign remaining majority clusters to closest subdomains
        unassigned_maj_clusters = set(range(self.maj_centroids.shape[0])) - assigned_maj_clusters
        
        print(f"Subdomains before assigning unassigned majority clusters: {subdomains_emp}")
        print(f"Unassigned majority clusters: {unassigned_maj_clusters}")
        
        for maj_cluster in unassigned_maj_clusters:
            # find closest minority centroid
            distances_to_min = pairwise_distances(self.maj_centroids[maj_cluster].reshape(1, -1), self.min_centroids).flatten()
            closest_min_idx = np.argmin(distances_to_min)
            
            # find the subdomain with this minority centroid and add the majority cluster
            for subdomain in subdomains_emp:
                if subdomain[0] == closest_min_idx:
                    subdomain.append(maj_cluster)
                    break
                
        # form subdomain and fit MotherNetClassifier on each subdomain
        print(f"Subdomain: {subdomains_emp}")
        print("*"*50)
        
        #check position of majority clusters closest to each other
        min_cluster_distances = pairwise_distances(self.min_centroids,self.min_centroids)
        closest_min_clusters = np.argsort(min_cluster_distances, axis=1)
        
        #sort subdomains (list of lists) based on the closest majority cluster distances to the first subdomain
        subdomains_emp = np.array(subdomains_emp,dtype=object)
        subdomains_emp = subdomains_emp[closest_min_clusters[0,:]]
        subdomains_emp = subdomains_emp.tolist()
                
        subdomain_idx = 0
        final_subdomain_idx = 0
        pairs = {}
        Total_sum = 0
        size_thr = 2000
        
        if self.verbose:
            print(f"X shape: {X.shape}, Number of empirical subdomains formed: {len(subdomains_emp)}")
        
        # first generate 1 subdomain and fit a model, then generate next subdomain and see if all samples are correctly classified, 
        # merge subdomains and fit a new model, else if a samples is missclasified, keep previous subdomain separate and fit a new model for the new subdomain
        while subdomain_idx < len(subdomains_emp):
            
            pairs[final_subdomain_idx] = []
            
            if self.verbose:
                print(f"Processing subdomain {subdomain_idx}")
            
            #generate subdomain data
            X_sub, y_sub, boolean_subdomain = self.generate_empirical_subdomains(subdomains_emp[subdomain_idx],X,y)
            
            #fit model for subdomain
            if self.classifier_type == 'MotherNet':
                
                model = MotherNetClassifier(device='cuda',inference_device='cuda')
            else:
                #model = TabPFNClassifier.create_default_for_version(ModelVersion.V2)
                model = MLPClassifier(hidden_layer_sizes=(512,512), max_iter=1000, random_state=42,alpha=0.1)
                #model = MotherNetClassifier(device='cuda',inference_device='cuda')
            model.fit(X_sub, y_sub)
            
            perfect_loss = True
            # set a limit of merged subdomains
            correct_counter = 0

            size = X_sub.shape[0]
            
            while perfect_loss and (subdomain_idx + 1) < len(subdomains_emp) and size < size_thr:
                
                if self.verbose:
                    print(f"Trying to merge subdomain {subdomain_idx} and {subdomain_idx+1}")
                
                #generate next subdomain data
                X_sub_next, y_sub_next, boolean_subdomain_next = self.generate_empirical_subdomains(subdomains_emp[subdomain_idx+1],X,y)
         
                
                if self.verbose:
                    print(f"Shape and class distribution of next subdomain: {X_sub_next.shape}, {np.unique(y_sub_next,return_counts=True)}")

                #predict on merged subdomain
                preds = model.predict(X_sub_next)
                
                #check if all samples are correctly classified
                incorrect_indices = np.where(preds != y_sub_next)[0]
                
                if len(incorrect_indices) < (1/(1000*correct_counter+1))*len(y_sub_next):
                    
                    #all samples are correctly classified, merge subdomains                
                    #merge current and next subdomain
                    X_merged = np.vstack([X_sub, X_sub_next])
                    y_merged = np.hstack([y_sub, y_sub_next])
                    
                    #Merge boolean trackers where if a sample belongs to either subdomain, it is marked as True
                    boolean_subdomain = np.logical_or(boolean_subdomain, boolean_subdomain_next)
                
                    X_sub = X_merged
                    y_sub = y_merged
                    subdomain_idx += 1
                    size = X_sub.shape[0]
                    correct_counter += 1
                    if self.verbose:
                        print(f"Size after merge: {size}, size_thr: {size_thr}")
                        print(f"Merged subdomain {subdomain_idx} successfully. New shape: {X_sub.shape}, {np.unique(y_sub,return_counts=True)}")
                        
                    #fit model on merged subdomain
                    if self.classifier_type == 'MotherNet':
                        model = MotherNetClassifier(device='cuda',inference_device='cuda')
                    else:
                        #model = TabPFNClassifier.create_default_for_version(ModelVersion.V2)
                        model = MLPClassifier(hidden_layer_sizes=(512,512), max_iter=1000, random_state=42,alpha=0.1)
                        #model = MotherNetClassifier(device='cuda',inference_device='cuda')
                    model.fit(X_sub, y_sub)
                    
                    pairs[final_subdomain_idx].append(subdomain_idx)
                    
                else:
                    perfect_loss = False
                    if self.verbose:
                        print(f"Could not merge subdomain {subdomain_idx+1}, {len(incorrect_indices)} misclassified samples.")

            #save trained model for current subdomain
            centroid = np.median(X_sub, axis=0)
            self.trained_models_empirical_[f"Model_{final_subdomain_idx}"] = [centroid,model]
            self.domain_indices_on_X_empirical[final_subdomain_idx] = boolean_subdomain
            subdomain_idx += 1
            final_subdomain_idx += 1
            Total_sum += np.sum(boolean_subdomain)
            
            if self.verbose:
                print(f"Trained subdomain model Model_{final_subdomain_idx-1} with {np.sum(boolean_subdomain)} samples.")
                
        print(f"Merging percentage of subdomains: {(final_subdomain_idx)/(len(subdomains_emp))*100:.2f}%")
        print(f"Total samples covered by empirical subdomains: {Total_sum} out of {X.shape[0]} samples.")     
        
    def fit_mixture_hypernetworks(self, X, y):
        
        #get classwise stats and clusters
        self.compute_classwise_stats(y)
        self.compute_classwise_cluster(X,y)
        self.get_subdomains()
        self.n_training_samples_ = X.shape[0]
        
        # iteratively generate subdomains and fit MotherNetClassifier on each subdomain
        self.trained_models_ = {}
        self.trained_models_empirical_ = {}
        self.domain_indices_on_X = {i : np.zeros((len(y),)) for i in range(len(self.subdomains))}
        self.domain_indices_on_X_empirical = {}
        
        #check position of majority clusters closest to each other
        majority_cluster_distances = pairwise_distances(self.maj_centroids,self.maj_centroids)
        closest_maj_clusters = np.argsort(majority_cluster_distances, axis=1)[:,1:]
        
        for i, subdomain in enumerate(self.subdomains):
            
            #get data for subdomain
            idx = np.flatnonzero(y == self.class_counts_["maj"]["class"])
            
            X_maj_sub = X[idx[self.maj_clusters == subdomain[0]]]
            
            # try to mark domain indices for majority samples in subdomain
            self.domain_indices_on_X[i][idx[self.maj_clusters == subdomain[0]]] = 1
            
            
            if self.verbose:
                print(f"Initial number of majority samples in cluster {subdomain[0]}: {X_maj_sub.shape[0]}")
                print(f"Shape: {self.domain_indices_on_X[i].shape} | Domain indices sum for subdomain {i} before majority addition: {np.sum(self.domain_indices_on_X[i])} should match {X_maj_sub.shape[0]} samples.")
            
            #--------------------------
            # code for overlap 
            
            #target final number of majority samples
            X_maj_sub_target = int(np.round(X_maj_sub.shape[0]*(1+self.overlap_thr)))
            
            #append samples from closest majority clusters until X_maj_sub.shape[0] exceeds X_min_sub.shape[0]*(1+self.overlap_thr)
            for m, maj_centroid_idx in enumerate(closest_maj_clusters[subdomain[0]]):
                
                # if majority samples exceed X_maj_sub.shape[0]*self.domain_thr, compute difference and select the required number of samples
                if X_maj_sub.shape[0] > X_maj_sub_target:
                    diff = X_maj_sub.shape[0] - X_maj_sub_target
                    X_maj_sub = X_maj_sub[:-diff,:]
                    
                    #remove indices previously added from domain_indices_on_X, not that maj_centroid_idx is not the original cluster
                    self.domain_indices_on_X[i][idx[self.maj_clusters == closest_maj_clusters[subdomain[0]][m-1]][-diff:]] = 0
                    
                    break
                
                else:
                    
                    #subdomain indices are marked as 1 for samples added to X_maj_sub
                    self.domain_indices_on_X[i][idx[self.maj_clusters == maj_centroid_idx]] = 1
                    X_maj_sub = np.vstack([X_maj_sub, X[idx[self.maj_clusters == maj_centroid_idx]]])
                    
            #--------------------------

            if self.verbose:
                print(f"Initial number of majority samples in cluster {subdomain[0]}: {X_maj_sub.shape[0]}")
                print(f"Domain indices sum for subdomain {i} after majority addition: {np.sum(self.domain_indices_on_X[i])} should match {X_maj_sub.shape[0]} samples.")
                   
            
            X_min_sub = []
            count =  0

            idx_min  = np.flatnonzero(y == self.class_counts_["min"]["class"])

            #iterate over minority centroids in subdomain and add samples until matching majority samples
            for m, min_centroid_idx in enumerate(subdomain[1:]):
                    
                indexes = self.min_clusters == min_centroid_idx
                
                count += np.sum(indexes)
                X_min_sub.append(X[idx_min[indexes]])
                
                #mark domain indices for minority samples as 1
                self.domain_indices_on_X[i][idx_min[indexes]] = 1
                
                # if majority samples exceed X_min_sub.shape[0]*self.domain_thr, randomly sample to match the last majority cluster
                if  count > int(np.round(X_maj_sub.shape[0])):
                    diff = count - int(np.round(X_maj_sub.shape[0]))
                    X_min_sub[-1] = X_min_sub[-1][:-diff,:]
                    
                    #remove indices previously added from domain_indices_on_X
                    self.domain_indices_on_X[i][idx_min[self.min_clusters == subdomain[1:][m-1]][-diff:]] = 0
                    
                    if self.verbose:
                        print(f"Majority size: {X_maj_sub.shape[0]} | Minority samples: {np.vstack(X_min_sub).shape[0]}")
                    
                    break
                
            # if nimber of minority samples is less than majority samples, randomly subsample majority samples to match
            total_minority_samples = np.vstack(X_min_sub).shape[0]
            if total_minority_samples < X_maj_sub.shape[0]:
                diff = X_maj_sub.shape[0] - total_minority_samples
                rand_indices = np.random.choice(np.arange(X_maj_sub.shape[0]), size=diff, replace=False)
                X_maj_sub = np.delete(X_maj_sub, rand_indices, axis=0)
                
                #remove indices previously added from domain_indices_on_X
                self.domain_indices_on_X[i][idx[rand_indices]] = 0
                
                if self.verbose:
                    print(f"After sampling, Minority samples: {np.vstack(X_min_sub).shape[0]} to match Majority samples: {X_maj_sub.shape[0]}")

            X_min_sub = np.vstack(X_min_sub)
            X_sub = np.vstack([X_min_sub, X_maj_sub])
            y_sub = np.hstack([np.full(X_min_sub.shape[0], self.class_counts_["min"]["class"]),
                               np.full(X_maj_sub.shape[0], self.class_counts_["maj"]["class"])])
            mediod = np.median(X_sub, axis=0)


            if self.verbose:
                print(f"Fitting subdomain {i+1}/{len(self.subdomains)} with {X_maj_sub.shape[0]} majority and {X_min_sub.shape[0]} minority samples.")
                print(f"Domain indices sum for subdomain {i}: {np.sum(self.domain_indices_on_X[i])} should match {X_sub.shape[0]} samples.")
            
            #fit models for all subdomains for balanced
            if self.classifier_type == 'MotherNet':
                model = MotherNetClassifier(device='cuda',inference_device="cuda")
            else:
                #model = TabPFNClassifier.create_default_for_version(ModelVersion.V2)
                model = MLPClassifier(hidden_layer_sizes=(512,512), max_iter=1000, random_state=self.random_state,alpha=0.1)
                #model = MotherNetClassifier(device='cuda',inference_device='cuda')
            model.fit(X_sub, y_sub)
            self.trained_models_[f"Model_{i}"] = [mediod,model]
        
        #print number of trained models
        if self.verbose:
            print(f"Number of trained subdomain models: {len(self.trained_models_)}") 
            
            
        if X.shape[0] < 3000:
        # if self.class_counts_["min"]["count"] < 500:
            
            max_samples = np.min([10000,X.shape[0]])
            if self.classifier_type == 'MotherNet':
                base_model = MotherNetClassifier(device='cuda',inference_device='cuda')
                #model = BaggingClassifier(estimator=base_model, n_estimators=10, random_state=self.random_state,max_samples= max_samples)
                #base_model = TabPFNClassifier.create_default_for_version(ModelVersion.V2)
            else:
                #model = TabPFNClassifier.create_default_for_version(ModelVersion.V2)
                base_model = MLPClassifier(hidden_layer_sizes=(512,512), max_iter=1000, random_state=self.random_state,alpha=0.1)
            model = BaggingClassifier(estimator=base_model, n_estimators=10, random_state=self.random_state,max_samples= max_samples)
            
            model.fit(X, y)
            self.trained_models_empirical_['Model_0'] = [np.mean(X, axis=0),model]
            boolean = np.ones((len(y),))
            self.domain_indices_on_X_empirical[0] = boolean
            
        
        else:
            # train meta models to predict best model for each subdomain
            self.fit_empirical_mixture_hypernetworks(X,y)
          
            print(f"Number of trained empirical subdomain models: {len(self.trained_models_empirical_)}")
            
        self.train_meta_models(X)
        

    def predict_2(self, X_test, decision_threshold=0.7):

        n_samples = X_test.shape[0]
        if self.verbose:
            print(f"Predicting {n_samples} samples with decision threshold: {decision_threshold}")

        #if X_test array dtype is not float32, convert with assaray
        # if X_test.dtype != np.float32:
        #     X_test = np.asarray(X_test, dtype=np.float32)
        #     print(f"Converted X_test to float32 dtype.")
            
        # ---------------------------------------------------------
        # 1. Meta-Model Predictions
        # ---------------------------------------------------------
        meta_predictions_balanced = np.zeros((n_samples, len(self.meta_models_balanced_)))
        meta_predictions_empirical = np.zeros((n_samples, len(self.meta_models_empirical_)))
        
        # Get meta model predictions (Balanced)
        for i, key in enumerate(self.meta_models_balanced_.keys()):
            model_balanced = self.meta_models_balanced_[key]
            preds_balanced = model_balanced.predict_proba(X_test)
            meta_predictions_balanced[:, i] = preds_balanced[:, 1]
            
        # Get meta model predictions (Empirical)
        for i, key in enumerate(self.meta_models_empirical_.keys()):
            model_empirical = self.meta_models_empirical_[key]
            preds_empirical = model_empirical.predict_proba(X_test)
            if len(self.meta_models_empirical_) > 1:
                meta_predictions_empirical[:, i] = preds_empirical[:, 1]
            else:
                meta_predictions_empirical[:, i] = np.ones((n_samples,))

        # ---------------------------------------------------------
        # 2. Schedule Batches (Determine which samples go to which model)
        # ---------------------------------------------------------
        
        # Helper to compute distances efficiently for fallback
        # Returns matrix of shape (n_samples, n_models)
        def compute_distances(models_dict, n_models):
            # Extract centroids in order
            centroids = np.array([models_dict[f"Model_{j}"][0] for j in range(n_models)])
            
            if self.verbose:
                print(f"Computing distances for {n_samples} samples and {n_models} models.")
                print(f"Centroids shape: {centroids.shape}| centroids: {centroids} | dtype: {centroids.dtype}")
                print(f"X_test shape: {X_test.shape}| dtype: {X_test.dtype}")
            
            # Vectorized distance: sqrt(sum((x - y)^2))
            # using broadcasting: (N, 1, F) - (1, M, F) -> (N, M, F)
            # Note: If memory is an issue with (N, M, F), we can compute in chunks, 
            # but usually n_models is small enough.
            diff = X_test[:, np.newaxis, :] - centroids[np.newaxis, :, :]
            
            if self.verbose:
                print(f"Diff shape: {diff.shape}| dtype: {diff.dtype}")
            
            return np.linalg.norm(diff, axis=2)

        # Pre-calculate distances for fallback scenarios
        # We only compute this if we suspect we need it, but calculating it once is cleaner 
        # than doing it per-sample in a loop.
        n_models_bal = len(self.meta_models_balanced_)
        n_models_emp = len(self.meta_models_empirical_)
        
        dists_balanced_all = compute_distances(self.trained_models_, n_models_bal)
        dists_empirical_all = compute_distances(self.trained_models_empirical_, n_models_emp)

        # batch_queues: model_idx -> {'sample_indices': [], 'weights': []}
        batch_queues_balanced = defaultdict(lambda: {'indices': [], 'weights': []})
        batch_queues_empirical = defaultdict(lambda: {'indices': [], 'weights': []})
        
        is_meta_balanced = np.zeros(n_samples, dtype=bool)
        is_meta_empirical = np.zeros(n_samples, dtype=bool)

        # --- Assign Balanced Models ---
        for i in range(n_samples):
            # Check threshold
            sample_preds = meta_predictions_balanced[i, :]
            above_threshold_idx = np.where(sample_preds >= decision_threshold)[0]
            
            if len(above_threshold_idx) > 0:
                # Meta-model selection
                above_threshold_probs = sample_preds[above_threshold_idx]
                top_k = min(3, len(above_threshold_idx))
                # Argsort is ascending, take last k and reverse
                top_indices_local = np.argsort(above_threshold_probs)[-top_k:][::-1]
                
                selected_idx = above_threshold_idx[top_indices_local]
                selected_probs = above_threshold_probs[top_indices_local]
                is_meta_balanced[i] = True
            else:
                # Fallback: Distance based
                dists = dists_balanced_all[i]
                closest_indices = np.argsort(dists)[:2]
                selected_idx = closest_indices
                
                selected_dists = dists[closest_indices]
                eps = 1e-8
                selected_probs = 1.0 / (selected_dists + eps)
                is_meta_balanced[i] = False
            
            # Queue the requests
            for model_idx, prob in zip(selected_idx, selected_probs):
                batch_queues_balanced[model_idx]['indices'].append(i)
                batch_queues_balanced[model_idx]['weights'].append(prob)

        # --- Assign Empirical Models ---
        for i in range(n_samples):
            sample_preds = meta_predictions_empirical[i, :]
            above_threshold_idx = np.where(sample_preds >= decision_threshold)[0]
            
            if len(above_threshold_idx) > 0:
                above_threshold_probs = sample_preds[above_threshold_idx]
                top_k = min(3, len(above_threshold_idx))
                top_indices_local = np.argsort(above_threshold_probs)[-top_k:][::-1]
                
                selected_idx = above_threshold_idx[top_indices_local]
                selected_probs = above_threshold_probs[top_indices_local]
                is_meta_empirical[i] = True
            else:
                dists = dists_empirical_all[i]
                closest_indices = np.argsort(dists)[:2]
                selected_idx = closest_indices
                
                selected_dists = dists[closest_indices]
                eps = 1e-8
                selected_probs = 1.0 / (selected_dists + eps)
                is_meta_empirical[i] = False
                
            for model_idx, prob in zip(selected_idx, selected_probs):
                batch_queues_empirical[model_idx]['indices'].append(i)
                batch_queues_empirical[model_idx]['weights'].append(prob)

        # ---------------------------------------------------------
        # 3. Execute Batches & Aggregate
        # ---------------------------------------------------------
        
        predictions_balanced = np.zeros((n_samples, 2))
        weights_sum_balanced = np.zeros(n_samples)
        
        predictions_empirical = np.zeros((n_samples, 2))
        weights_sum_empirical = np.zeros(n_samples)

        # --- Run Balanced Batches ---
        for model_idx, batch_data in batch_queues_balanced.items():
            indices = batch_data['indices']
            weights = np.array(batch_data['weights'])
            
            # Fetch model [1] is the model, [0] was centroid
            mothernet_model = self.trained_models_[f"Model_{model_idx}"][1]
            
            # Batch prediction!
            X_batch = X_test[indices]
            preds_batch = mothernet_model.predict_proba(X_batch) # Shape (n_batch, 2)
            
            # Weighted accumulation
            # weights is (n_batch,), preds_batch is (n_batch, 2) -> broadcast
            weighted_preds = preds_batch * weights[:, np.newaxis]
            
            predictions_balanced[indices] += weighted_preds
            weights_sum_balanced[indices] += weights

        # --- Run Empirical Batches ---
        for model_idx, batch_data in batch_queues_empirical.items():
            indices = batch_data['indices']
            weights = np.array(batch_data['weights'])
            
            mothernet_model = self.trained_models_empirical_[f"Model_{model_idx}"][1]
            
            X_batch = X_test[indices]
            preds_batch = mothernet_model.predict_proba(X_batch)
            
            weighted_preds = preds_batch * weights[:, np.newaxis]
            
            predictions_empirical[indices] += weighted_preds
            weights_sum_empirical[indices] += weights

        # ---------------------------------------------------------
        # 4. Finalize
        # ---------------------------------------------------------
        
        # Normalize weighted averages
        # Avoid division by zero (though logic ensures at least fallback models are selected)
        mask_bal = weights_sum_balanced > 0
        predictions_balanced[mask_bal] /= weights_sum_balanced[mask_bal, np.newaxis]
        
        mask_emp = weights_sum_empirical > 0
        predictions_empirical[mask_emp] /= weights_sum_empirical[mask_emp, np.newaxis]

        # Apply scaling logic for small datasets where meta model wasn't used
        # Note: We check if the LAST model used was meta or not in original logic. 
        # Here we tracked is_meta per sample.
        scale_mask_bal = (~is_meta_balanced) & (self.class_counts_["min"]["count"] < 500)
        predictions_balanced[scale_mask_bal] /= 10**6

        # Original code commented this out, but keeping structure if needed:
        # scale_mask_emp = (~is_meta_empirical) & (is_meta_balanced) & (self.n_training_samples_ < 2000)
        # predictions_empirical[scale_mask_emp] /= 10**6

        # Merge predictions
        alpha = self.alpha
        combined_preds = (alpha * predictions_balanced + (1 - alpha) * predictions_empirical)
        predictions_final = np.argmax(combined_preds, axis=1)

        return None, None, predictions_final, predictions_balanced, predictions_empirical, alpha, combined_preds
    
class CustomEnsembleClassifier:
    def __init__(self, base_model_class, model_params, n_estimators=10, mode='bagging', random_state=42):
        """
        Args:
            base_model_class: The class constructor for the base model (e.g., MixtureHypernetworks).
            model_params (dict): Dictionary of parameters to initialize the base model.
            n_estimators (int): Number of estimators in the ensemble.
            mode (str): One of 'regular', 'bagging', 'adaboost'.
            random_state (int): Seed for reproducibility.
        """
        self.base_model_class = base_model_class
        self.model_params = model_params
        self.n_estimators = n_estimators
        self.mode = mode.lower()
        self.random_state = random_state
        
        self.estimators_ = []
        self.estimator_weights_ = []
        self.classes_ = None
        
        # Valid modes check
        if self.mode not in ['regular', 'bagging', 'adaboost']:
            raise ValueError(f"Unknown mode: {self.mode}. Choose 'regular', 'bagging', or 'adaboost'.")

    def fit(self, X, y):
        """
        Fits the ensemble based on the selected mode.
        """
        X = np.array(X)
        y = np.array(y)
        n_samples = X.shape[0]
        self.classes_ = np.unique(y)
        
        # Initialize sample weights for AdaBoost (uniform initially)
        # These represent the probability of being sampled
        sample_weights = np.ones(n_samples) / n_samples
        
        rng = np.random.RandomState(self.random_state)
        
        # Clear previous fits
        self.estimators_ = []
        self.estimator_weights_ = []

        for i in range(self.n_estimators):
            
            print(f"Training estimator {i+1}/{self.n_estimators} in mode '{self.mode}'")
            
            # 1. Determine Training Data based on Mode
            if self.mode == 'regular':
                # Mode 1: Regular Ensemble
                # Train on all data, just vary the random state of the model
                indices = np.arange(n_samples)
                oob_indices = [] # No OOB in regular mode unless we force a split
                
            elif self.mode == 'bagging':
                # Mode 2: Bagging
                # Sample with replacement (uniform probability)
                indices = resample(np.arange(n_samples), replace=True, random_state=rng)
                
            elif self.mode == 'adaboost':
                # Mode 3: AdaBoost via Manual Weighted Resampling
                # Normalize weights to ensure they sum to exactly 1.0 for numpy
                weight_sum = np.sum(sample_weights)
                if weight_sum > 0:
                    p_distribution = sample_weights / weight_sum
                else:
                    # Fallback to uniform if weights collapse
                    p_distribution = np.ones(n_samples) / n_samples
                
                # Manual weighted sampling using numpy choice
                indices = rng.choice(
                    np.arange(n_samples), 
                    size=n_samples, 
                    replace=True, 
                    p=p_distribution
                )
            
            # Identify OOB indices for weighting logic
            # (In 'regular' mode, this set is empty)
            all_indices = np.arange(n_samples)
            oob_indices = np.setdiff1d(all_indices, indices)
            
            # 2. Initialize and Fit Base Model
            # Vary random state per estimator if the base model accepts it
            current_params = self.model_params.copy()
            if 'random_state' in current_params: 
                current_params['random_state'] = rng.randint(0, 10000)
                print(f"Setting 'random_state' to {current_params['random_state']} for estimator {i}")
            if 'm' in current_params:
                current_params['m'] = rng.randint(2, 8) if current_params['random_state'] % 2 == 0 else rng.randint(2, 8) 
                print(f"Setting 'm' to {current_params['m']} for estimator {i}")
            # if 'majority_cluster' in current_params:
            #     # can be either KMeans or BalancedKMeans
            #     if current_params['random_state'] % 2 == 0:
            #         current_params['majority_cluster'] = 'KMeans'
            #     else:
            #         current_params['majority_cluster'] = 'BalancedKMeansLSA'
            if 'minority_cluster' in current_params:
                
                if current_params['random_state'] % 2 == 0:
                    current_params['minority_cluster'] = 'KMeans'
                else:
                    current_params['minority_cluster'] = 'BalancedKMeansLSA'
            else:
                print(f"No random state parameter found for base model; using default settings.")
                
            model = self.base_model_class(**current_params)
            
            X_train = X[indices].astype(np.float32)
            y_train = y[indices]
            
            # --- USER SPECIFIC FIT CALL ---
            model.fit_mixture_hypernetworks(X_train, y_train)
            
            # 3. Calculate Ensemble Weight (OOB Estimate)
            if len(oob_indices) > 0:
                # Predict on OOB data to get weight
                X_oob = X[oob_indices].astype(np.float32)
                y_oob = y[oob_indices]
                
                # --- USER SPECIFIC PREDICT CALL ---
                _, _, y_pred_oob, _, _, _, _ = model.predict_2(X_oob)
                
                acc = accuracy_score(y_oob, y_pred_oob)
                # Avoid zero weight; add small epsilon or use raw acc
                weight = acc if acc > 0 else 1e-5
            else:
                # Fallback for 'regular' mode or if bootstrap captured all samples (rare)
                weight = 1.0 
            
            self.estimators_.append(model)
            self.estimator_weights_.append(weight)
            
            # 4. AdaBoost Step: Update sample probabilities for NEXT iteration
            if self.mode == 'adaboost':
                # We need predictions on the WHOLE dataset to update weights based on error
                # Note: This uses predict_2 as defined by user
                _, _, y_pred_all, _, _, _, _ = model.predict_2(X.astype(np.float32))
                
                # Calculate error (incorrect predictions)
                incorrect = (y_pred_all != y).astype(float)
                
                # Weighted error rate
                # (epsilon) = sum(weights of misclassified) / sum(weights)
                estimator_error = np.dot(incorrect, sample_weights) / np.sum(sample_weights)
                
                # Safety checks for division by zero or perfect fit
                if estimator_error >= 1.0 - 1e-10:
                    estimator_error = 1.0 - 1e-10
                if estimator_error <= 1e-10:
                    estimator_error = 1e-10
                
                # Calculate Alpha (amount of say this model would usually have, 
                # though we are using OOB for final voting as requested)
                alpha = 0.5 * np.log((1.0 - estimator_error) / estimator_error)
                
                # Update sample weights: increase weight if incorrect
                # w_new = w_old * exp(alpha * incorrect) 
                # (incorrect is 1 if wrong, 0 if right. If right, weight stays same? 
                # Standard Adaboost decreases weight for correct ones. 
                # Let's use standard formula: w * exp(-alpha * y * h(x)) equivalent)
                
                # Simplified update for multiclass/resampling:
                # If incorrect, multiply by exp(alpha). If correct, multiply by exp(-alpha).
                multipliers = np.where(incorrect == 1, np.exp(alpha), np.exp(-alpha))
                sample_weights *= multipliers
                
                # Normalize weights to sum to 1
                sample_weights /= np.sum(sample_weights)

    def predict(self, X):
        """
        Predict class labels for X.
        """
        X = np.array(X).astype(np.float32)
        
        # Soft voting storage
        # Shape: (n_samples, n_classes)
        # Assuming classes are 0, 1, ..., n_classes-1 for simplicity
        # If classes are strings/arbitrary, we'd map them.
        n_classes = len(self.classes_)
        sum_proba = np.zeros((X.shape[0], n_classes))
        sum_weights = 0.0
        
        for model, weight in zip(self.estimators_, self.estimator_weights_):
            # --- USER SPECIFIC PREDICT CALL ---
            # Returns: _, _, y_pred, y_proba, percent_meta, is_meta, compression_rate
            #_, _, _, y_proba, _, _, _ = model.predict_2(X)
            _, _, _, _, _, _, y_proba = model.predict_2(X)
            
            # Accumulate weighted probabilities
            sum_proba += (y_proba * weight)
            sum_weights += weight
            
        # Normalize
        avg_proba = sum_proba / sum_weights
        
        # Argmax to get class labels
        predicted_indices = np.argmax(avg_proba, axis=1)
        return self.classes_[predicted_indices]

    def predict_proba(self, X):
        """
        Returns weighted average probabilities.
        """
        X = np.array(X).astype(np.float32)
        n_classes = len(self.classes_)
        sum_proba = np.zeros((X.shape[0], n_classes))
        sum_proba_balanced = np.zeros((X.shape[0], n_classes))
        sum_proba_empirical = np.zeros((X.shape[0], n_classes))
        sum_weights = 0.0
        
        for model, weight in zip(self.estimators_, self.estimator_weights_):
            #_, _, _, y_proba, _, _, _ = model.predict_2(X)
            _, _, _, y_balanced, y_empirical, _, y_proba = model.predict_2(X)
            sum_proba += (y_proba * weight)
            sum_proba_balanced += (y_balanced * weight)
            sum_proba_empirical += (y_empirical * weight)
            sum_weights += weight
            

            
        return sum_proba / sum_weights, sum_proba_balanced / sum_weights, sum_proba_empirical / sum_weights
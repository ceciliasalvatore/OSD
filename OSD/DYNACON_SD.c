//
// Code for the DYNACON_SD algorithm
//

#include <stdio.h>
#include <stdlib.h>

struct DYNACON_node{
    struct DYNACON_node *predecessor;
    int i;
    int *A;
    int *V;
    int l;
    int T;
    double objective;
};

struct DYNACON_node init_node(int i, int *A, struct DYNACON_node *source, int n, int m, double lambda){
    struct DYNACON_node node;
    node.predecessor = source;
    node.l = (i>-1 & i<n)? 1:0;
    node.T = node.l;
    node.i = i;
    node.A = (i>-1 & i<n)? &(A[m*i]):NULL;
    node.V = calloc(m, sizeof(int));
    if (node.V == NULL) {
        perror("Error in calloc.\n");
    }
    int count = m;
    if (node.A != NULL){
        for (int j=0; j<m; j++){
            node.V[j] = node.A[j];
            count -= node.V[j];
        }
    }
    node.objective = (i>-1 & i<n)? node.T + count*lambda : node.T+m*lambda*10;
    return node;
}

void compare_predecessors(struct DYNACON_node *node, struct DYNACON_node *p, int m, double lambda){
    int *V_p = calloc(m, sizeof(int));
    int T_p = p->T+node->l;

    if (V_p==NULL){
        perror("Error in calloc\n");
    }
    int count = m;
    if (node->A != NULL) {
        for (int i = 0; i < m; i++) {
            V_p[i] = (node->A)[i] | (p->V)[i];
            count -= V_p[i];
        }
    }
    else {
        for (int i = 0; i < m; i++) {
            V_p[i] = (p->V)[i];
            count -= V_p[i];
        }
    }
    double objective = T_p+count*lambda;
    if (objective < node->objective){
        node->predecessor = p;
        node->T = T_p;
        free(node->V);
        node->V = V_p;
        node->objective = objective;
    }
    else {
        free(V_p);
    }
}

int DYNACON_SD(int n, int m, int *A, double lambda, int *I, double *objective) {
    struct DYNACON_node *nodes = malloc((n + 2) * sizeof(struct DYNACON_node));
    if (nodes == NULL) {
        perror("Error in malloc.\n");
        return 1;
    }
    nodes[0] = init_node(-1, A, NULL, n, m, lambda);

    for (int i = 0; i < n + 1; i++) {
        nodes[i+1] = init_node(i, A, &nodes[0], n, m, lambda);
        for (int j = 0; j < i; j++) {
            compare_predecessors(&nodes[i+1], &nodes[j+1], m, lambda);
        }
    }
    struct DYNACON_node *node = &nodes[n+1];
    while (node != NULL) {
        if (node->i>-1 & node->i<n) {
            I[node->i] = 1;
        }
        node = node->predecessor;
    }
    *objective = nodes[n+1].objective;//node_objective(nodes[n+1],m,lambda);
    for (int i=-1; i<n+1; i++) {
        free(nodes[i+1].V);
    }
    free(nodes);
    return 0;
}
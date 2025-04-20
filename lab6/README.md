## Large Scale Computing (Lab 6)
### Jakub Płowiec

This exercise was done with the minikube on Ubuntu.

1. Basic installation

    a. Minikube
    ```
    curl -LO https://github.com/kubernetes/minikube/releases/latest/download/minikube-linux-amd64
    sudo install minikube-linux-amd64 /usr/local/bin/minikube && rm minikube-linux-amd64
    ```

    b. Kubectl
    ```
    curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
    sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
    ```

2. Create cluster
    ```
    minikube start
    ```

3. Apply configuration
    ```
    kubectl apply -f pvc.yaml
    kubectl apply -f nginx-deployment.yaml
    kubectl apply -f nginx-service.yaml
    kubectl apply -f job.yaml
    ```

4. Start service
    ```bash
    minikube service nginx-service
    ```
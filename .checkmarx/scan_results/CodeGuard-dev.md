
# CodeGuard Scan Report:  Sanofi-OneAI/oneai-com-turing-causal_inference <br/> 
 ![img](https://img.shields.io/badge/SCA%20-%201%20HIGH%20vuln.%20found-red.svg) ![img](https://img.shields.io/badge/SAST%20-%20NO%20vuln.%20found-green.svg)

__Source:__ zip   --   __Branch:__ dev  
__Scan Execution date:__ 2024-08-22T06:55:35.451551Z  
__Scan Id:__ cf1b7728-84ee-498b-b375-a6175a84caae  
__Scan Status:__ Completed
## Executive summary
You'll find below a list of vulnerabilities identified by our service on both Static Application Security Testing and Software Composition Analysis dimensions:
  - SCA analyzes open source and 3rd party libraries (vulnerabilities and legal risks)
  - SAST focuses on custom code (built by the development team)

### Vulnerabilities:
 <table align="center" >
  <tr>
    <th colspan="2"> :red_circle: NO-GO for production :red_circle: </th>
    <th> :large_orange_diamond: to be remediated within 90 days :large_orange_diamond: </th>
  </tr>
  <tr>    <th>SCA</th>
    <th>SAST</th>
    <th>SAST</th>
  </tr>
  <tr align="center">
    <td>1</td>
    <td>0</td>
    <td>0</td>

  </tr>
</table>


### Legal Risks:
1 high issues found.
Please ensure you understand and agree with packages license terms before deploying to production.


#####

💡 If you need some help on this report, please [submit an issue](https://github.com/Sanofi-Shared-GitHub-Apps/CodeGuardSupport/issues/new?template=support_form.yml&title=Code+Guard+support+form&scan-id=cf1b7728-84ee-498b-b375-a6175a84caae&scan-branch=dev&URL=Sanofi-OneAI/oneai-com-turing-causal_inference) in our CodeGuardSupport repo.

#####

## :red_circle: NO-GO for production alerts 

### SCA Alerts

#### List of Packages impacted by HIGH CVEs (direct and transitive)

 <details><summary><b>
Python-pyro-ppl-1.9.1
 </b></summary><blockquote>

location: [requirements.txt](/Sanofi-OneAI/oneai-com-turing-causal_inference/blob/dev/requirements.txt)

 <details><summary>vulnerabilities inherited from  <b> Python-torch-2.4.0 </b> 

 </summary><blockquote>

   <details><summary>CVE-2024-5480 </summary><blockquote>
A vulnerability in PyTorch's "torch.distributed.rpc" framework allows for Remote Code Execution (RCE). The framework, which is used in distributed training scenarios, does not properly verify the functions being called during RPC (Remote Procedure Call) operations. This oversight permits attackers to execute arbitrary commands by leveraging built-in Python functions such as "eval" during multi-cpu RPC communication. The vulnerability arises from the lack of restriction on function calls when a worker node serializes and sends a PythonUDF (User Defined Function) to the master node, which then deserializes and executes the function without validation. This flaw can be exploited to compromise master nodes initiating distributed training, potentially leading to the theft of sensitive AI-related data.

[more info](https://devhub.checkmarx.com/cve-details/CVE-2024-5480)
 </blockquote></details>
 </blockquote></details>
 </blockquote></details>


 <details><summary><b>
Python-torch-2.4.0
 </b></summary><blockquote>

location: [requirements.txt](/Sanofi-OneAI/oneai-com-turing-causal_inference/blob/dev/requirements.txt)

 <details><summary>direct vulnerabilities found
 </summary><blockquote>

   <details><summary>CVE-2024-5480 </summary><blockquote>
A vulnerability in PyTorch's "torch.distributed.rpc" framework allows for Remote Code Execution (RCE). The framework, which is used in distributed training scenarios, does not properly verify the functions being called during RPC (Remote Procedure Call) operations. This oversight permits attackers to execute arbitrary commands by leveraging built-in Python functions such as "eval" during multi-cpu RPC communication. The vulnerability arises from the lack of restriction on function calls when a worker node serializes and sends a PythonUDF (User Defined Function) to the master node, which then deserializes and executes the function without validation. This flaw can be exploited to compromise master nodes initiating distributed training, potentially leading to the theft of sensitive AI-related data.

[more info](https://devhub.checkmarx.com/cve-details/CVE-2024-5480)
 </blockquote></details>
 </blockquote></details>
 </blockquote></details>



#### List of Container Packages with HIGH CVE
NO Container Packages detected


#### List of Packages with HIGH Legal risks
 Packages detected 

_Please read carefully the license terms associated with these packages, and check [Sanofi open-source policy](https://docs.sanofi.com/cpv/wiki/spaces/OSPO/pages/64038509339#Usingopen-sourcelibraries&packages-Open-sourcelicenses).  
 In case of any doubt, contact the Open-Source Program Office (ospo@sanofi.com)._  
   <details><summary><b>numpy - 1.26.4 </b></summary><blockquote>

  location: [requirements.txt](/Sanofi-OneAI/oneai-com-turing-causal_inference/blob/dev/requirements.txt)

   <details><summary>origin:  </summary><blockquote> 
  
  - Python-matplotlib-3.9.2 > Python-numpy-1.26.4
  - Python-numpy-1.26.4
  - Python-pandas-2.2.2 > Python-numpy-1.26.4
  - Python-prophet-1.1.5 > Python-numpy-1.26.4
  - Python-pyro-ppl-1.9.1 > Python-numpy-1.26.4
  - Python-seaborn-0.13.2 > Python-numpy-1.26.4
  - Python-tensorflow-2.17.0 > Python-numpy-1.26.4
  - Python-tensorflow-probability-0.24.0 > Python-numpy-1.26.4
  - Python-tf-keras-2.17.0 > Python-tensorflow-2.17.0 > Python-numpy-1.26.4 </blockquote></details>


  __Risks:__

  - Issue Name: AGPL 3.0 - Copyright risk score: 7 - Patent risk score: 1 - Copyleft: Full </blockquote></details>


### HIGH SAST Alerts

NO alert

## :large_orange_diamond: to be remediated within 90 days

### MEDIUM SAST Alerts
NO alert

---------------------------

### SAST issues remediation estimate

Based on the type and the number of SAST alerts identified above, an estimated workload to fix them is __less than 1 day__ of development.


// SPDX-License-Identifier: MIT
pragma solidity ^0.8.4; 

contract DeepfakeDetection {

    
    mapping(bytes32 => uint256) private _videoHashToResult;

   
    event VideoResultUpdated(bytes32 indexed videoHash, uint256 result);

    
    function getVideoResult(bytes32 videoHash) public view returns (uint256) {
        return _videoHashToResult[videoHash];
    }

    
    function updateVideoResult(bytes32 videoHash, uint256 newResult) public {
        
        uint256 cachedResult = _videoHashToResult[videoHash];

        
        if (cachedResult != newResult) {
            _videoHashToResult[videoHash] = newResult;

           
            emit VideoResultUpdated(videoHash, newResult);
        }
    }


    function isVideoDetected(bytes32 videoHash) public view returns (bool) {
  
        bool detectionResult = _videoHashToResult[videoHash] > 0;
        
        return detectionResult; 
    }
}

//----------------------------------------------------------------------------------------------------------------------
// This declarative Jenkins pipeline encodes all the steps required for the nightly/continuous of a single platform.
// Other jobs may call this pipeline to execute the build, test and installation of a set platforms.
//
// Author: Pere Mato
//----------------------------------------------------------------------------------------------------------------------

pipeline {
  parameters {
    string(name: 'EXTERNALS', defaultValue: 'devgeantv/latest', description: 'LCG software stack in CVMFS')
    choice(name: 'MODE', choices: ['experimental', 'nightly', 'continuous'], description: 'CDash mode')
    string(name: 'ExtraCMakeOptions', defaultValue: '', description: 'CMake extra configuration options')
    string(name: 'LABEL', defaultValue: 'centos7', description: 'Jenkins label for physical nodes or container image for docker')
    choice(name: 'LABEL', choices: ['Release', 'Debug'])
    string(name: 'PLATFORM', defaultValue: 'x86_64+avs2+fma-centos7-gcc9-opt', description: 'The Platform of the stack to be used')
    choice(name: 'OPTION', choices: ['default', 'SPEC', 'AVX', 'GDML'])
    choice(name: 'BACKEND', choices: ['scalar', 'vc'])
    string(name: 'DOCKER_LABEL', defaultValue: 'docker-host-noafs', description: 'Label for the the nodes able to launch docker images')
    string(name: 'SourceBranch', defaultValue: 'master', description: 'Source branch in repository')
    string(name: 'TargetBranch', defaultValue: 'master', description: 'Target branch in repository')
    string(name: 'gitlabMergedByUser')
    string(name: 'gitlabMergeRequestIid')
  }

  environment {
    CMAKE_INSTALL_PREFIX = 'install'
    CMAKE_SOURCE_DIR     = 'vecgeom'
    CMAKE_BINARY_DIR     = 'build'
  }

  agent none

  stages {
    //------------------------------------------------------------------------------------------------------------------
    //---Build & Test stages--------------------------------------------------------------------------------------------
    //------------------------------------------------------------------------------------------------------------------
    stage('Prepa'){
      steps {
        init()
      }
    }
    stage('InDocker') {
      when {
        beforeAgent true
        expression { params.LABEL =~ 'centos|ubuntu' && !(params.LABEL =~ 'physical')}
      }
      agent {
        docker {
          image "gitlab-registry.cern.ch/sft/docker/$LABEL"
          label "$DOCKER_LABEL"
          args  """-v /cvmfs:/cvmfs 
                   -v /ccache:/ccache 
                   -v /ec:/ec
                   -e SHELL 
                   -e gitlabMergedByUser 
                   -e gitlabMergeRequestIid
                   --net=host
                   --hostname ${LABEL}-docker
                """
        }
      }
      stages {
        stage('Build&Test') {
          steps {
            buildAndTest()
          }
          post {
            success {
              deleteDir()
            }
          }
        }
      }
    }
    stage('InBareMetal') {
      when {
        beforeAgent true
        expression { params.LABEL =~ 'cuda|physical' }
      }
      agent {
        label "$LABEL"
      }
      stages {
        stage('Build&Test') {
          steps {
            buildAndTest()
          }
          post {
            success {
              deleteDir()
            }
          }
        }
      }
    }
  }
}

def init() {
  currentBuild.displayName = "#${BUILD_NUMBER}" + ' ' + params.OPTION + '-' + params.BACKEND + '-' + params.LABEL + '-' + params.PLATFORM + '-' + params.BUILDTYPE
}

def buildAndTest() {
  sh label: 'build_and_test', script: """
    if [[ ${LABEL} =~ cuda ]]; then
        source /cvmfs/sft.cern.ch/lcg/contrib/cuda/11.4/x86_64-centos7/setup.sh
    fi
    source /cvmfs/sft.cern.ch/lcg/views/${EXTERNALS}/${PLATFORM}/setup.sh
    env | sort | sed 's/:/:?     /g' | tr '?' '\n'
    ctest -VV -S vecgeom/jenkins/vecgeom-ctest.cmake,$MODE
  """
}

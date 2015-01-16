import os,sys,glob,datetime
import numpy as np

from nipype.interfaces import spm, fsl, afni, nitime, utility, lif, dcmstack as np_dcmstack, freesurfer, nipy

import nipype.pipeline.engine as pe
import nipype.interfaces.io as nio
import nibabel as nb

fsl.FSLCommand.set_default_output_type('NIFTI_GZ')
afni.base.AFNICommand.set_default_output_type('NIFTI_GZ')
np_dcmstack.DCMStackBase.set_default_output_type('NIFTI_GZ')

sys.path.insert(0,'/home/bpinsard/data/src/misc')
import generic_pipelines
from generic_pipelines.utils import wrap, fname_presuffix_basename, wildcard

from nipype import config
cfg = dict(execution={'stop_on_first_crash': False})
config.update_config(cfg)

data_dir = '/home/bpinsard/data/raw/UNF/CoRe'
proc_dir = '/home/bpinsard/data/analysis/'

subjects = ['S40_JB']



tr=2.16
file_pattern = '_%(PatientName)s_%(SeriesDate)s_%(SeriesTime)s'
meta_tag_force=['PatientID','PatientName','SeriesDate']

def dicom_dirs():

    subjects_info = pe.Node(
        utility.IdentityInterface(fields=['subject']),
        name='subjects_info')
    subjects_info.iterables = [('subject', subjects)]

    dicom_dirs = pe.Node(
        nio.DataGrabber(infields=['subject'],
                        outfields=['t1_mprage','aa_scout','localizer',
                                   'fmri_resting_state','fmri_task1','fmri_task2',
                                   'fmri_fieldmap', 'fmri_pa'],
                        sort_filelist = True,
                        base_directory = data_dir, template=''),
        name='dicom_dirs')
    dicom_dirs.inputs.template = 'CoRe_%s_*/??-%s'
    dicom_dirs.inputs.template_args = dict(
        t1_mprage=[['subject','anat_mprage-MPRAGE_12ch']],
        aa_scout=[['subject','AAScout']],
        localizer=[['subject','localizer_12Channel']],
        fmri_resting_state=[['subject','fmri-BOLD_Resting_State']],
        fmri_task1=[['subject','fmri-BOLD_Task1']],
        fmri_task2=[['subject','fmri-BOLD_Task2']],
        fmri_fieldmap=[['subject','fmri_fieldmap-gre_field_map_BOLD']],
        fmri_pa=[['subject','fmri-BOLD_PA']],)

    w=pe.Workflow(name='core_pilot')

    for n in [dicom_dirs]:
        w.connect([(subjects_info,n,[('subject',)*2])])
        
    return w
    
def preproc_anat():
    
    w = dicom_dirs()

    n_t1_convert = pe.Node(
        np_dcmstack.DCMStackAnatomical(
            meta_force_add=meta_tag_force,
            out_file_format = 't1_mprage'+file_pattern,
            voxel_order='LAS'),
        name='convert_t1_dicom')

    n_fieldmap_fmri_convert = pe.MapNode(
        np_dcmstack.DCMStackFieldmap(
            meta_force_add=meta_tag_force,
            out_file_format = 'fieldmap_bold'+file_pattern,
            voxel_order='LAS'),
        iterfield = ['dicom_files'],
        name='convert_fieldmap_dicom')

    t1_pipeline = generic_pipelines.t1_new.t1_freesurfer_pipeline()
    t1_pipeline.inputs.freesurfer.args=''
    wm_surface = generic_pipelines.t1_new.extract_wm_surface()

    n_crop_t1 = pe.Node(
        nipy.Crop(out_file='%s_crop.nii.gz',
                  outputtype='NIFTI_GZ'),
        name='crop_t1')
    t1_pipeline.connect([
            (t1_pipeline.get_node('autobox_mask_fs'),n_crop_t1,
             [('%s_%s'%(c,p),)*2 for c in 'xyz' for p in ('min','max')]),
            (t1_pipeline.get_node('freesurfer'),n_crop_t1,
             [('norm','in_file')])])

    n_flirt_fmap2t1 = pe.MapNode(
        fsl.FLIRT(cost='mutualinfo',
                  dof=6,
                  out_matrix_file = 'fmap2t1.mat',
                  no_search=True,
                  uses_qform=True),
        iterfield=['in_file'],
        name='flirt_fmap2t1')

    n_flirt2xfm = pe.MapNode(
        freesurfer.preprocess.Tkregister(reg_file='reg.mat',
                                         xfm_out='fmap2t1.xfm'),
        iterfield=['mov','fsl_reg'],
        name='flirt2xfm')

    n_xfm2mat = pe.MapNode(
        utility.Function(
            input_names = ['xfm'],
            output_names = ['mat'],
            function='def f(xfm): import numpy as np; mat=np.loadtxt(xfm, skiprows=5,usecols=range(4)); np.savetxt("out.mat",np.vstack([mat,[0,0,0,1]]));return "out.mat"'),
        iterfield=['xfm'],
        name='xfm2mat')
    
    n_low_res_surf = pe.Node(
        freesurfer.SurfaceTransform(
            target_subject='ico',
            target_ico_order=6,),
        name='low_res_surf')

    n_low_res_surf.iterables = [('hemi',['lh','rh']),
                                ('source_surface_file',['white','pial'])]

    n_low_res_parc = pe.Node(
        freesurfer.MRIConvert(vox_size=(2.,)*3,out_type='niigz',
                              args='-rt nearest'),
        name='low_res_parc')
    n_reg_crop = pe.Node(
        freesurfer.Tkregister(reg_file='reg_crop.dat',reg_header=True),
        'reg_crop')
    n_compute_pvmaps = pe.Node(
        freesurfer.ComputeVolumeFractions(),
        name='compute_pvmaps')

    t1_pipeline.connect([
            (t1_pipeline.get_node('freesurfer'),n_low_res_surf,
             [('subjects_dir','subjects_dir'),
              ('subject_id','source_subject')]),
            (t1_pipeline.get_node('freesurfer'),n_low_res_parc, [
                    (('aparc_aseg',utility.select,1), 'in_file')]),
            (t1_pipeline.get_node('freesurfer'),n_reg_crop,[
                    ('norm','target'),('subject_id',)*2,('subjects_dir',)*2]),
            (t1_pipeline.get_node('autobox_mask_fs'),n_reg_crop,[
                    ('out_file','mov')]),
            (n_reg_crop, n_compute_pvmaps, [
                    ('reg_file',)*2]),
            (t1_pipeline.get_node('autobox_mask_fs'), n_compute_pvmaps, [
                    ('out_file','in_file')]),
            (t1_pipeline.get_node('freesurfer'),n_compute_pvmaps,[
                    ('subjects_dir',)*2]),            
            ])
    
    w.base_dir = proc_dir
    w.connect([
            (w.get_node('dicom_dirs'),n_t1_convert,[('t1_mprage','dicom_files')]),
            (w.get_node('subjects_info'),t1_pipeline,[('subject','inputspec.subject_id')]),
            (n_t1_convert,t1_pipeline,[('nifti_file','inputspec.t1_file')]),
            (t1_pipeline, wm_surface, [(('freesurfer.aparc_aseg',utility.select,0),'inputspec.aseg')]),
       ])

    return w


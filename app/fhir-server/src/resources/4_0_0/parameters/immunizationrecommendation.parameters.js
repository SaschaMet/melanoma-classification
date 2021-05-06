const { GraphQLString } = require('graphql');
const DateScalar = require('../scalars/date.scalar.js');
const TokenScalar = require('../scalars/token.scalar.js');

/**
 * @name exports
 * @static
 * @summary Arguments for the immunizationrecommendation query
 */
module.exports = {
	// http://hl7.org/fhir/SearchParameter/ImmunizationRecommendation-date
	date: {
		type: DateScalar,
		fhirtype: 'date',
		xpath: 'ImmunizationRecommendation.date',
		description: 'Date recommendation(s) created',
	},
	// http://hl7.org/fhir/SearchParameter/ImmunizationRecommendation-identifier
	identifier: {
		type: TokenScalar,
		fhirtype: 'token',
		xpath: 'ImmunizationRecommendation.identifier',
		description: 'Business identifier',
	},
	// http://hl7.org/fhir/SearchParameter/ImmunizationRecommendation-information
	information: {
		type: GraphQLString,
		fhirtype: 'reference',
		xpath:
			'ImmunizationRecommendation.recommendation.supportingPatientInformation',
		description: 'Patient observations supporting recommendation',
	},
	// http://hl7.org/fhir/SearchParameter/ImmunizationRecommendation-patient
	patient: {
		type: GraphQLString,
		fhirtype: 'reference',
		xpath: 'ImmunizationRecommendation.patient',
		description: 'Who this profile is for',
	},
	// http://hl7.org/fhir/SearchParameter/ImmunizationRecommendation-status
	status: {
		type: TokenScalar,
		fhirtype: 'token',
		xpath: 'ImmunizationRecommendation.recommendation.forecastStatus',
		description: 'Vaccine recommendation status',
	},
	// http://hl7.org/fhir/SearchParameter/ImmunizationRecommendation-support
	support: {
		type: GraphQLString,
		fhirtype: 'reference',
		xpath: 'ImmunizationRecommendation.recommendation.supportingImmunization',
		description: 'Past immunizations supporting recommendation',
	},
	// http://hl7.org/fhir/SearchParameter/ImmunizationRecommendation-target-disease
	target_disease: {
		type: TokenScalar,
		fhirtype: 'token',
		xpath: 'ImmunizationRecommendation.recommendation.targetDisease',
		description: 'Disease to be immunized against',
	},
	// http://hl7.org/fhir/SearchParameter/ImmunizationRecommendation-vaccine-type
	vaccine_type: {
		type: TokenScalar,
		fhirtype: 'token',
		xpath: 'ImmunizationRecommendation.recommendation.vaccineCode',
		description: 'Vaccine  or vaccine group recommendation applies to',
	},
};
